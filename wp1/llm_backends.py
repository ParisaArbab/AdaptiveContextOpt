"""
llm_backends.py — one provider layer for every model the study compares.

FlexFL's whole premise is that it works with open-source models, not just
frontier APIs, so the backend layer has to treat "a 32B Qwen on your own GPU"
and "gpt-4o over the network" as the same kind of thing. Everything here
resolves to the same contract the rest of the pipeline already speaks:

    chat_fn(system: str, user: str) -> str

Three provider kinds cover the field:

  openai_compatible — the de-facto standard. OpenAI, DeepSeek, Qwen
      (DashScope), Ollama (local AND cloud), vLLM, LM Studio, TGI,
      llama.cpp's server, OpenRouter, Together, Groq, Mistral, Gemini's
      compatibility endpoint. These differ only in base_url, key env var,
      and default model — so they are registry rows, not code.
  anthropic — its own SDK and message shape.
  heuristic — no model at all; the key-free stand-in used to validate
      plumbing offline.

Anything not in the registry still works:
    --llm custom --base-url http://my-box:8000/v1 --model my/model

Three things open-source models need that hosted ones mostly don't, all
handled here rather than sprinkled through the localizer:

  1. REASONING TAGS. DeepSeek-R1, QwQ, Qwen3-thinking and friends emit
     `<think>...</think>` before the answer. FlexFL parses `Top_1 : ...`
     lines and one `FunctionName(Argument)` per turn, and a reasoning block
     containing the word "Top_1" or a function call breaks both. Stripped
     from every response, plus the `reasoning_content` field DeepSeek's API
     returns out-of-band.
  2. RETRIES. Local servers 503 while a model loads; hosted ones rate-limit.
     A single dropped call would otherwise abort a whole ReAct loop and lose
     an instance from every arm.
  3. CONTEXT-LIMIT ERRORS have to stay recognisable, because FlexFL's
     adaptive-MAX behaviour (paper §3.2.1) keys off them to shrink the loop
     and retry. Retry logic deliberately re-raises those instead of
     swallowing them — a context overflow is not a transient failure.
"""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

ChatFn = Callable[[str, str], str]

# Markers agent_localizer's adaptive-MAX loop keys off. Kept here so the
# retry wrapper and the loop agree on what "the context is too long" means.
CONTEXT_ERROR_MARKERS = (
    "context length", "context_length", "maximum context", "too many tokens",
    "token limit", "reduce the length", "context window", "prompt is too long",
    "input length and `max_tokens`",
)

RETRYABLE_MARKERS = (
    "rate limit", "rate_limit", "429", "500", "502", "503", "504",
    "overloaded", "timeout", "timed out", "connection", "temporarily",
    "service unavailable", "model is loading", "please try again",
)

_THINK_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|reflection)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
# An unterminated opener happens when a reasoning model hits max_tokens
# mid-thought. Everything after it is reasoning, not answer.
_THINK_OPEN_RE = re.compile(r"<(think|thinking|reasoning|reflection)>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks so FlexFL's parsers see only the answer."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


def is_context_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in CONTEXT_ERROR_MARKERS)


def is_retryable(exc: BaseException) -> bool:
    if is_context_error(exc):
        return False  # a real limit, not a blip — let adaptive MAX handle it
    msg = str(exc).lower()
    return any(m in msg for m in RETRYABLE_MARKERS)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

@dataclass
class ProviderSpec:
    name: str
    kind: str                       # "openai_compatible" | "anthropic" | "heuristic"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    default_model: Optional[str] = None
    requires_key: bool = True
    open_source: bool = False       # serves open-weight models
    local: bool = False             # runs on your own hardware
    note: str = ""


PROVIDERS: Dict[str, ProviderSpec] = {
    # -- hosted, proprietary ------------------------------------------------
    "openai": ProviderSpec("openai", "openai_compatible", "https://api.openai.com/v1",
                           "OPENAI_API_KEY", "gpt-4o"),
    "anthropic": ProviderSpec("anthropic", "anthropic", None,
                              "ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
    "gemini": ProviderSpec("gemini", "openai_compatible",
                           "https://generativelanguage.googleapis.com/v1beta/openai/",
                           "GEMINI_API_KEY", "gemini-2.0-flash"),
    # -- hosted, open-weight models ----------------------------------------
    "deepseek": ProviderSpec("deepseek", "openai_compatible", "https://api.deepseek.com",
                             "DEEPSEEK_API_KEY", "deepseek-chat", open_source=True),
    "qwen": ProviderSpec("qwen", "openai_compatible",
                         "https://dashscope.aliyuncs.com/compatible-mode/v1",
                         "DASHSCOPE_API_KEY", "qwen-max", open_source=True),
    "openrouter": ProviderSpec("openrouter", "openai_compatible",
                               "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                               None, open_source=True,
                               note="model ids are namespaced, e.g. qwen/qwen-2.5-coder-32b-instruct"),
    "together": ProviderSpec("together", "openai_compatible", "https://api.together.xyz/v1",
                             "TOGETHER_API_KEY", None, open_source=True),
    "groq": ProviderSpec("groq", "openai_compatible", "https://api.groq.com/openai/v1",
                         "GROQ_API_KEY", None, open_source=True),
    "mistral": ProviderSpec("mistral", "openai_compatible", "https://api.mistral.ai/v1",
                            "MISTRAL_API_KEY", "mistral-large-latest", open_source=True),
    "ollama-cloud": ProviderSpec("ollama-cloud", "openai_compatible", "https://ollama.com/v1",
                                 "OLLAMA_API_KEY", None, open_source=True,
                                 note="Ollama's hosted models; same model ids as local ollama"),
    # -- local, your own hardware ------------------------------------------
    "ollama": ProviderSpec("ollama", "openai_compatible", "http://localhost:11434/v1",
                           "OLLAMA_API_KEY", None, requires_key=False,
                           open_source=True, local=True,
                           note="run `ollama serve`; model = whatever `ollama list` shows"),
    "vllm": ProviderSpec("vllm", "openai_compatible", "http://localhost:8000/v1",
                         "VLLM_API_KEY", None, requires_key=False,
                         open_source=True, local=True,
                         note="`vllm serve <repo-id>`; model = the repo id you served"),
    "lmstudio": ProviderSpec("lmstudio", "openai_compatible", "http://localhost:1234/v1",
                             None, None, requires_key=False, open_source=True, local=True),
    "tgi": ProviderSpec("tgi", "openai_compatible", "http://localhost:8080/v1",
                        None, None, requires_key=False, open_source=True, local=True,
                        note="text-generation-inference in OpenAI-compatible mode"),
    "llamacpp": ProviderSpec("llamacpp", "openai_compatible", "http://localhost:8080/v1",
                             None, None, requires_key=False, open_source=True, local=True,
                             note="llama.cpp's `llama-server`"),
    # -- escape hatches -----------------------------------------------------
    "custom": ProviderSpec("custom", "openai_compatible", None, None, None,
                           requires_key=False,
                           note="any OpenAI-compatible server; pass --base-url and --model"),
    "heuristic": ProviderSpec("heuristic", "heuristic", requires_key=False,
                              note="key-free deterministic stand-in, no model involved"),
}

# Spellings people reach for.
ALIASES: Dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "openai-compatible": "custom",
    "dashscope": "qwen",
    "local": "vllm",          # back-compat with the earlier --backend local
    "google": "gemini",
    "none": "heuristic",
}


@dataclass
class LLMConfig:
    provider: str = "heuristic"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0        # deterministic by default: arms must be comparable
    max_tokens: int = 1024
    timeout: float = 300.0
    max_retries: int = 4
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @property
    def spec(self) -> ProviderSpec:
        return PROVIDERS[ALIASES.get(self.provider, self.provider)]

    @property
    def resolved_model(self) -> Optional[str]:
        return self.model or self.spec.default_model

    @property
    def label(self) -> str:
        """Goes into results as the backend name — provider AND model, since
        'deepseek' alone doesn't identify what was actually run."""
        if self.spec.kind == "heuristic":
            return "heuristic"
        return f"{self.spec.name}:{self.resolved_model or 'unset'}"


def list_providers() -> str:
    rows = []
    width = max(len(n) for n in PROVIDERS)
    for name, spec in PROVIDERS.items():
        tags = []
        if spec.local:
            tags.append("local")
        if spec.open_source:
            tags.append("open-weights")
        if not spec.requires_key:
            tags.append("no-key")
        rows.append(f"  {name:<{width}}  {spec.default_model or '(model required)':<28}"
                    f"  {','.join(tags) or '-'}"
                    + (f"\n  {'':<{width}}  {spec.note}" if spec.note else ""))
    alias_line = "  aliases: " + ", ".join(f"{a}->{t}" for a, t in ALIASES.items())
    return "\n".join(rows) + "\n" + alias_line


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _with_retries(call: Callable[[], str], cfg: LLMConfig) -> str:
    last: Optional[BaseException] = None
    for attempt in range(cfg.max_retries):
        try:
            return call()
        except Exception as e:
            if is_context_error(e):
                raise               # adaptive MAX must see this, not a retry
            if not is_retryable(e) or attempt == cfg.max_retries - 1:
                raise
            last = e
            # exponential backoff with jitter; a local server loading a 32B
            # model can take tens of seconds before it answers at all
            time.sleep(min(30.0, (2 ** attempt) + random.random()))
    raise RuntimeError(f"exhausted retries: {last}")


def _make_openai_compatible(cfg: LLMConfig) -> ChatFn:
    try:
        import openai
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("`pip install openai` is required for this provider") from e

    spec = cfg.spec
    base_url = cfg.base_url or spec.base_url
    key_env = cfg.api_key_env or spec.api_key_env
    api_key = os.environ.get(key_env) if key_env else None
    if not api_key:
        if spec.requires_key:
            raise RuntimeError(
                f"{spec.name} needs an API key. Set {key_env}, or pass "
                f"--api-key-env <VAR> if you keep it somewhere else."
            )
        # Local servers ignore the field but the SDK requires it non-empty.
        api_key = "local"
    model = cfg.resolved_model
    if not model:
        raise RuntimeError(
            f"{spec.name} has no default model — pass --model. "
            + (spec.note or "")
        )

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=cfg.timeout)

    def chat_fn(system: str, user: str) -> str:
        def call() -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                extra_headers=cfg.extra_headers or None,
            )
            choice = resp.choices[0].message
            text = choice.content or ""
            # DeepSeek's reasoner returns thinking out-of-band; some gateways
            # inline it instead. Drop both, keep only the answer.
            if not text and getattr(choice, "reasoning_content", None):
                text = choice.reasoning_content or ""
            return strip_reasoning(text)

        return _with_retries(call, cfg)

    return chat_fn


def _make_anthropic(cfg: LLMConfig) -> ChatFn:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("`pip install anthropic` is required for this provider") from e

    key_env = cfg.api_key_env or cfg.spec.api_key_env
    if key_env and not os.environ.get(key_env):
        raise RuntimeError(f"anthropic needs {key_env} to be set")
    client = anthropic.Anthropic(timeout=cfg.timeout)
    model = cfg.resolved_model

    def chat_fn(system: str, user: str) -> str:
        def call() -> str:
            msg = client.messages.create(
                model=model, max_tokens=cfg.max_tokens, system=system,
                temperature=cfg.temperature,
                messages=[{"role": "user", "content": user}],
            )
            return strip_reasoning("".join(b.text for b in msg.content if hasattr(b, "text")))

        return _with_retries(call, cfg)

    return chat_fn


def build_chat_fn(cfg: LLMConfig) -> Optional[ChatFn]:
    """Returns None for the heuristic backend, which has no model."""
    kind = cfg.spec.kind
    if kind == "heuristic":
        return None
    if kind == "anthropic":
        return _make_anthropic(cfg)
    if kind == "openai_compatible":
        return _make_openai_compatible(cfg)
    raise ValueError(f"unknown provider kind {kind!r}")


def resolve(provider: str, **kwargs) -> LLMConfig:
    key = ALIASES.get(provider, provider)
    if key not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r}. Known providers:\n{list_providers()}"
        )
    return LLMConfig(provider=key, **{k: v for k, v in kwargs.items() if v is not None})


def preflight(cfg: LLMConfig) -> str:
    """Cheap round-trip so a 6-arm x 15-instance run doesn't die on call #1.
    Returns a human-readable status line; raises if the provider is
    unusable."""
    if cfg.spec.kind == "heuristic":
        return "heuristic backend — no model, no credentials needed"
    chat_fn = build_chat_fn(cfg)
    reply = chat_fn("Reply with the single word: ready", "ready?")
    return f"{cfg.label} reachable (replied {reply.strip()[:40]!r})"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="list or test LLM providers")
    ap.add_argument("--check", metavar="PROVIDER", help="preflight a provider")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    args = ap.parse_args()

    if args.check:
        cfg = resolve(args.check, model=args.model, base_url=args.base_url,
                      api_key_env=args.api_key_env)
        print(preflight(cfg))
    else:
        print("providers:\n" + list_providers())
