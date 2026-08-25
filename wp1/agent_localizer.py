"""
agent_localizer.py — WP1

Hybrid localization agent combining the two methods specified for this
project:
  - FlexFL: two-stage ReAct localization (originally Java/Defects4J here
    adapted to Python/SWE-bench) — coarse file-level pass, then fine
    function-level pass within candidate files.
  - GraphLocator: symptom-vertex locating + RDFS neighbor expansion — here,
    the "graph" GraphLocator expands over is the Graphify structure map
    (wp1/graphify_structure.py) instead of GraphLocator's original RDFS
    graph, since Graphify already gives us a local, real, queryable
    code graph per instance.

Model-agnostic by design: this is the exact component whose behavior gets
compared across Claude, GPT, DeepSeek, and Qwen per the project's stated
goal of evaluating closed vs. open models under compression.

Two backends:
  LLMBackend        — real chat-completions call, provider selected by
                       whichever *_API_KEY is set in the environment.
  HeuristicBackend   — deterministic, no API key needed. Used only to
                       validate the localization interface, scoring, and
                       feedback-loop wiring before credentials are available
                       — same role the earlier "deterministic heuristic
                       fallback" played for the rest of this project.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from feedback_loop import run_feedback_loop


@dataclass
class LocalizationResult:
    instance_id: str
    predicted_files: List[str]
    predicted_functions: List[str]
    backend: str
    feedback_rounds_used: int = 0


class LLMChatBackend(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class HeuristicBackend:
    """Deterministic, key-free stand-in for an LLM. Stage 1 (FlexFL-style
    coarse pass): score files by keyword overlap with the problem statement.
    Stage 2 (fine pass): pull function/class names out of the tool output's
    own stack-trace-looking lines. Stage 3 (GraphLocator-style expansion):
    pull in structure-map neighbors sharing the same 'community' as any
    stage-2 hit, as a stand-in for RDFS neighbor expansion.

    This is intentionally simple — it exists to prove the pipeline plumbing
    works end to end offline, not to produce a real localization result. Any
    accuracy numbers from this backend must be labeled backend='heuristic'
    and excluded from formal WP1 conclusions.
    """

    name = "heuristic"

    FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([A-Za-z_][A-Za-z0-9_]*)')

    def localize(self, tool_output: str, structure_map: dict, problem_statement: str) -> LocalizationResult:
        files, functions = set(), set()
        for m in self.FRAME_RE.finditer(tool_output):
            file_path, _line, func_name = m.groups()
            files.add(file_path)
            functions.add(f"{file_path}::{func_name}")

        # GraphLocator-style expansion: same-community neighbors of any hit
        hit_communities = {
            structure_map[f]["community"]
            for f in functions
            if f in structure_map
        }
        for key, meta in structure_map.items():
            if meta.get("community") in hit_communities and key not in functions:
                functions.add(key)
                files.add(meta["file"])

        return LocalizationResult(
            instance_id="",
            predicted_files=sorted(files),
            predicted_functions=sorted(functions),
            backend=self.name,
        )


def _select_llm_backend() -> Optional[str]:
    """Priority mirrors the project's stated model comparison: closed models
    first (Claude, GPT), then open models (DeepSeek, Qwen via any
    OpenAI-compatible endpoint)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"):
        return "qwen"
    return None


LOCALIZATION_SYSTEM_PROMPT = """You are a bug localization agent using a hybrid \
FlexFL + GraphLocator strategy. You are given: (1) a repository structure map \
(from Graphify, local AST-derived), (2) a problem statement, and (3) compressed \
or raw tool/test output. Stage 1 (FlexFL coarse): identify candidate files. \
Stage 2 (FlexFL fine): identify candidate functions/classes within those files. \
Stage 3 (GraphLocator expansion): use the structure map's community/graph \
neighbors of your stage-2 hits to check for related functions you may have missed. \
Respond ONLY with two lines:
FILES: comma-separated file paths
FUNCTIONS: comma-separated file::function_or_class entries"""

VERIFY_SYSTEM_PROMPT = """You just localized a bug from a possibly-compressed \
tool output. Given the current text, decide: does it contain enough evidence \
(stack trace, assertion diff, or failure signature) to trust your localization? \
If yes, respond exactly: OK
If something critical looks missing or truncated (e.g. an ellipsis where an \
assertion diff should be, a collapsed stack frame), respond exactly:
MISSING: L<start>-L<end> <one-sentence reason>
using your best estimate of the line range in the CURRENT text where something \
looks cut off."""


def _parse_localization(raw_response: str) -> tuple[List[str], List[str]]:
    files, functions = [], []
    for line in raw_response.splitlines():
        if line.upper().startswith("FILES:"):
            files = [f.strip() for f in line.split(":", 1)[1].split(",") if f.strip()]
        elif line.upper().startswith("FUNCTIONS:"):
            functions = [f.strip() for f in line.split(":", 1)[1].split(",") if f.strip()]
    return files, functions


def localize_with_llm(
    instance_id: str,
    tool_output: str,
    structure_map_text: str,
    problem_statement: str,
    chat_fn,  # Callable[[str, str], str] -> (system, user) -> response text
    backend_name: str,
    use_feedback_loop: bool = True,
    raw_tool_output: Optional[str] = None,
) -> LocalizationResult:
    """chat_fn is injected so this file has zero hard dependency on any one
    provider's SDK — plug in Anthropic, OpenAI, DeepSeek, or a Qwen
    OpenAI-compatible client here.

    `tool_output` is what the agent actually sees (raw, rtk-compressed, or
    lean-ctx-compressed, depending on which WP1 condition is running).
    `raw_tool_output` is only needed when use_feedback_loop=True and
    tool_output is a compressed variant — it's the cache the feedback loop
    reveals specific line ranges from. For the Control (raw) condition,
    leave raw_tool_output=None; the feedback loop then has nothing to
    reveal and effectively always resolves in round 1.
    """
    rounds_used = 0
    cache_text = raw_tool_output if raw_tool_output is not None else tool_output

    if use_feedback_loop:
        def verify_fn(current_text: str, round_num: int) -> str:
            return chat_fn(
                VERIFY_SYSTEM_PROMPT,
                f"CURRENT TEXT:\n{current_text}",
            )

        result = run_feedback_loop(
            raw_text=cache_text,
            compressed_text=tool_output,
            agent_verify_fn=verify_fn,
        )
        final_text = result.final_text
        rounds_used = result.rounds_used
    else:
        final_text = tool_output

    user_prompt = (
        f"STRUCTURE MAP:\n{structure_map_text}\n\n"
        f"PROBLEM STATEMENT:\n{problem_statement}\n\n"
        f"TOOL OUTPUT:\n{final_text}"
    )
    response = chat_fn(LOCALIZATION_SYSTEM_PROMPT, user_prompt)
    files, functions = _parse_localization(response)

    return LocalizationResult(
        instance_id=instance_id,
        predicted_files=files,
        predicted_functions=functions,
        backend=backend_name,
        feedback_rounds_used=rounds_used,
    )


def make_anthropic_chat_fn(model: str = "claude-sonnet-4-6"):
    import anthropic  # local import: only required if this backend is selected

    client = anthropic.Anthropic()

    def chat_fn(system: str, user: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))

    return chat_fn


def make_openai_compatible_chat_fn(
    api_key_env: str, base_url: Optional[str] = None, model: str = "gpt-4o"
):
    """Covers OpenAI, DeepSeek, and Qwen (via DashScope's OpenAI-compatible
    endpoint) with one function — all three speak the same chat-completions
    shape."""
    import openai  # local import

    client = openai.OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def chat_fn(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    return chat_fn
