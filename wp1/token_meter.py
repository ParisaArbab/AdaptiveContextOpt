"""
token_meter.py — WP1 step 2a: real token accounting.

The headline claim of this project is a token reduction, so token counts
cannot be a `len(text)//4` estimate and cannot be discarded after the
compressor runs. This module provides:

  TokenCounter — a real tokenizer where one is installed (tiktoken, or a
    HuggingFace tokenizer for open-source backends), falling back to the
    chars/4 estimate ONLY as a last resort and reporting `method` so every
    number downstream carries how it was measured.

  TokenMeter — wraps a chat_fn and accumulates prompt/completion tokens per
    PIPELINE STAGE (feedback / flexfl_stage1 / flexfl_stage2 / graph_expand),
    which is what makes "how many tokens did each element cost?" answerable
    directly instead of only by differencing ablation arms. It also records
    non-LLM context facts (raw vs compressed tool output size) in the same
    report so one object carries the whole token story for a run.

Stage attribution and arm differencing answer different questions and we
want both: staged counts say where tokens were spent inside one arm;
differencing across arms says what removing an element does to the total,
including its knock-on effect on how much the agent then has to search.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

# Stage labels — kept as constants so the analyzer and the localizer can't
# drift apart on spelling.
STAGE_FEEDBACK = "feedback"
STAGE_STAGE1 = "flexfl_stage1_agent4sr"
STAGE_STAGE2 = "flexfl_stage2_agent4lr"
STAGE_GRAPH = "graphlocator_expand"
STAGE_OTHER = "other"
ALL_STAGES = (STAGE_FEEDBACK, STAGE_STAGE1, STAGE_STAGE2, STAGE_GRAPH, STAGE_OTHER)


class TokenCounter:
    """Counts tokens with the best tokenizer available for the backend in use.

    Selection order:
      1. The served model's OWN HuggingFace tokenizer, when the model id
         looks like an HF repo (`Qwen/Qwen2.5-Coder-32B-Instruct`) and
         `transformers` is installed. For open-weight models this is the
         exact count the server itself computes, which matters because
         those are the models FlexFL is designed around.
      2. tiktoken `o200k_base`. Not byte-identical to Anthropic's or
         DeepSeek's tokenizer, but every arm is measured with the SAME one,
         so the cross-arm ratios — which is what the study reports — are
         sound even where the absolute count is a few percent off a given
         vendor's own billing count.
      3. chars/4. Last resort, and reported as such: `tokenizer_exact` goes
         false and the analyzer surfaces it instead of hiding it.

    Ollama-style tags (`qwen2.5-coder:32b`) are not HF repo ids, so they land
    on tiktoken — pass --tokenizer-model with the underlying HF id if you
    want exact counts for a locally-served model.
    """

    def __init__(self, provider: str = "heuristic", model: Optional[str] = None):
        self.provider = provider
        self.backend = provider          # retained: older result files read this
        self.model = model
        self.method = "chars_div_4_estimate"
        self._encode: Optional[Callable[[str], list]] = None
        self._hf = None

        candidate = model or os.environ.get("TOKENIZER_MODEL") or os.environ.get("LOCAL_LLM_MODEL")
        if candidate and "/" in candidate:
            self._try_hf(candidate)
        if self._encode is None and self._hf is None:
            self._try_tiktoken()

    def _try_tiktoken(self) -> None:
        try:
            import tiktoken
        except ImportError:
            return
        for enc_name in ("o200k_base", "cl100k_base"):
            try:
                enc = tiktoken.get_encoding(enc_name)
            except Exception:
                continue
            self._encode = enc.encode
            self.method = f"tiktoken:{enc_name}"
            return

    def _try_hf(self, model_id: Optional[str]) -> None:
        if not model_id:
            return
        try:
            from transformers import AutoTokenizer
            self._hf = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
            self.method = f"huggingface:{model_id}"
        except Exception:
            self._hf = None

    def count(self, text: Optional[str]) -> int:
        if not text:
            return 0
        if self._hf is not None:
            try:
                return len(self._hf.encode(text, add_special_tokens=False))
            except Exception:
                pass
        if self._encode is not None:
            try:
                return len(self._encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    @property
    def is_exact(self) -> bool:
        return self.method != "chars_div_4_estimate"


@dataclass
class StageUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenMeter:
    """Instruments a chat_fn. Usage:

        meter = TokenMeter(TokenCounter(backend))
        chat = meter.wrap(chat_fn)
        with meter.stage(STAGE_STAGE1):
            ...calls that use `chat`...
        meter.record_context("tool_output_raw", raw_text)
        report = meter.report()
    """

    def __init__(self, counter: TokenCounter):
        self.counter = counter
        self.stages: Dict[str, StageUsage] = {}
        self.context: Dict[str, int] = {}
        self._current = STAGE_OTHER

    @contextmanager
    def stage(self, name: str):
        previous = self._current
        self._current = name
        try:
            yield
        finally:
            self._current = previous

    def _bucket(self, name: str) -> StageUsage:
        return self.stages.setdefault(name, StageUsage())

    def wrap(self, chat_fn):
        """Returns a chat_fn with the identical (system, user) -> str contract
        that records both sides of every call against the active stage.
        Returns None if chat_fn is None, so the heuristic backend can pass
        straight through without a special case at the call site."""
        if chat_fn is None:
            return None

        def instrumented(system: str, user: str) -> str:
            bucket = self._bucket(self._current)
            bucket.calls += 1
            bucket.prompt_tokens += self.counter.count(system) + self.counter.count(user)
            response = chat_fn(system, user)
            bucket.completion_tokens += self.counter.count(response)
            return response

        return instrumented

    def record_context(self, label: str, text: Optional[str]) -> int:
        """Records a non-LLM token fact (e.g. the raw vs compressed tool
        output). These are what the compressor is directly acting on; the
        stage counts are what the agent then actually consumed."""
        n = self.counter.count(text)
        self.context[label] = n
        return n

    def report(self) -> dict:
        stages = {name: asdict(usage) | {"total_tokens": usage.total_tokens}
                  for name, usage in self.stages.items()}
        total_prompt = sum(u.prompt_tokens for u in self.stages.values())
        total_completion = sum(u.completion_tokens for u in self.stages.values())
        return {
            "tokenizer": self.counter.method,
            "tokenizer_exact": self.counter.is_exact,
            "llm_calls": sum(u.calls for u in self.stages.values()),
            "llm_prompt_tokens": total_prompt,
            "llm_completion_tokens": total_completion,
            "llm_total_tokens": total_prompt + total_completion,
            "by_stage": stages,
            "context_tokens": dict(self.context),
        }


def empty_report(counter: Optional[TokenCounter] = None) -> dict:
    return TokenMeter(counter or TokenCounter()).report()
