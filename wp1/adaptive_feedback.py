"""Gold-free feedback decisions for adaptive context expansion.

The feedback controller never receives fault-localization ground truth. It only
judges whether an Agent4SR run appears sufficiently supported by the runtime
context and the real Graphify tool evidence collected during that run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from wp1.llm_backends import ChatBackend


FEEDBACK_SYSTEM = """You are a context-sufficiency evaluator for software fault localization.

You are NOT the localization agent and you are NOT given the gold faulty file,
method, class, patch, or answer. Never guess or request gold information.

Decide only whether the current Agent4SR localization is sufficiently supported
by the evidence it actually inspected. Ask for more context when important
runtime evidence is missing, the reasoning is weak or conflicting, the agent did
not inspect enough relevant production code, or the final ranking is poorly
supported by the real Graphify results.

Return exactly two lines:
Decision: STOP
Reason: one short sentence

or:
Decision: EXPAND
Reason: one short sentence

STOP means keep the current localization result.
EXPAND means rerun localization with a higher LeanCTX density, which restores
more runtime context. Do not base the decision on whether the unknown gold
answer was found."""


@dataclass
class FeedbackDecision:
    decision: str
    reason: str
    raw_response: str
    source: str = "llm"

    @property
    def expand(self) -> bool:
        return self.decision == "EXPAND"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_feedback_decision(text: str) -> FeedbackDecision:
    match = re.search(r"(?im)^\s*Decision\s*:\s*(STOP|EXPAND)\s*$", text or "")
    reason_match = re.search(r"(?im)^\s*Reason\s*:\s*(.+?)\s*$", text or "")
    if not match:
        return FeedbackDecision(
            decision="EXPAND",
            reason="Feedback response was not parseable, so context expansion is used as the safe fallback.",
            raw_response=text or "",
            source="parse_fallback",
        )
    return FeedbackDecision(
        decision=match.group(1).upper(),
        reason=(reason_match.group(1).strip() if reason_match else "No reason supplied."),
        raw_response=text or "",
    )


def normalize_density_schedule(values: list[float] | tuple[float, ...]) -> list[float]:
    densities = sorted({round(float(value), 6) for value in values})
    if not densities:
        raise ValueError("density schedule cannot be empty")
    if densities[0] <= 0.0 or densities[-1] > 1.0:
        raise ValueError("densities must satisfy 0 < density <= 1")
    return densities


def next_density(current: float, schedule: list[float] | tuple[float, ...]) -> float | None:
    for density in normalize_density_schedule(schedule):
        if density > current + 1e-9:
            return density
    return None


def evaluate_context_sufficiency(
    backend: ChatBackend,
    *,
    problem: str,
    failing_test: str,
    runtime_output: str,
    agent_result: dict,
    current_density: float,
    next_density_value: float | None,
    feedback_round: int,
    max_feedback_rounds: int,
) -> FeedbackDecision:
    """Return a gold-free STOP/EXPAND decision for one Agent4SR run."""
    predictions = list(agent_result.get("predictions") or [])
    transcript = list(agent_result.get("transcript") or [])
    tools_used = set(agent_result.get("tools_used") or [])
    tool_calls = int(agent_result.get("tool_calls") or 0)

    if next_density_value is None:
        return FeedbackDecision(
            decision="STOP",
            reason="No higher density is available.",
            raw_response="",
            source="hard_stop",
        )

    if len(predictions) < 5:
        return FeedbackDecision(
            decision="EXPAND",
            reason="Agent4SR did not return five valid localization candidates.",
            raw_response="",
            source="hard_guard",
        )

    if tool_calls < 2 or "get_code_snippet" not in tools_used:
        return FeedbackDecision(
            decision="EXPAND",
            reason="Agent4SR did not collect enough real Graphify source evidence.",
            raw_response="",
            source="hard_guard",
        )

    history_lines: list[str] = []
    for index, item in enumerate(transcript[-8:], 1):
        assistant = str(item.get("assistant", ""))[:600]
        tool = str(item.get("tool", ""))[:1600]
        history_lines.append(
            f"Step {index} tool request:\n{assistant}\nActual Graphify result:\n{tool}"
        )

    prompt = f"""SWE-BENCH PROBLEM:
{problem}

FAILING TEST:
{failing_test}

CURRENT LEANCTX TARGET DENSITY: {current_density:.2f}
NEXT AVAILABLE DENSITY: {next_density_value:.2f}
FEEDBACK ROUND: {feedback_round}/{max_feedback_rounds}

CURRENT RUNTIME CONTEXT:
{runtime_output}

CURRENT AGENT4SR TOP-5:
""" + "\n".join(f"{i}. {p}" for i, p in enumerate(predictions, 1)) + "\n\nREAL TOOL EVIDENCE:\n" + "\n\n".join(history_lines)

    response = backend.complete(FEEDBACK_SYSTEM, prompt)
    return parse_feedback_decision(response)
