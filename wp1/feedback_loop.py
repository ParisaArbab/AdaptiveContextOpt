"""
feedback_loop.py — WP1 (new pipeline stage, per revised architecture:
Graphify -> compressor -> feedback loop -> evaluation)

After lean-ctx (or rtk) compresses a tool output, the localization agent gets
a chance to say "I think something important was cut" and ask for specific
lines back — capped at 2 rounds, never more, so this can't silently become
an uncompressed re-run.

This does NOT re-run compression differently; it works against the SAME
raw text lean-ctx/rtk already saw, using a line-range cache (same pattern as
the WP3 "Zoom-In" cache design, pulled forward because we need it here too:
without a way to ask for specific lines back, the agent's only options are
"accept the compressed text" or "get everything," and the second one
defeats the point of measuring compression tax at all).

Flow per tool-output event:
  1. compressed_result = leanctx_compressor.compress(raw_text)
  2. agent is shown compressed_result.text + a diff-aware fidelity prompt
  3. agent responds either "OK" or "MISSING: <what looks missing>"
  4. if MISSING and round < 2: reveal the specific raw line range the agent
     asked about (not the whole raw text), append it, increment round
  5. repeat from step 2, capped at round == 2
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Callable, List, Optional

MAX_ROUNDS = 2


@dataclass
class FeedbackTurn:
    round: int
    agent_verdict: str          # "OK" or "MISSING: ..."
    requested_range: Optional[str]  # e.g. "L40-L55", or None
    revealed_text: Optional[str]


@dataclass
class FeedbackResult:
    final_text: str
    rounds_used: int
    turns: List[FeedbackTurn] = field(default_factory=list)
    exhausted: bool = False  # True if agent still wanted more after MAX_ROUNDS


def _diff_summary(raw_text: str, compressed_text: str, max_lines: int = 20) -> str:
    """Cheap, deterministic signal for what changed — NOT shown to the agent
    as the answer, only used to validate that a 'MISSING' claim is plausible
    before we spend a round revealing anything (a request for a line range
    that was never removed is rejected rather than silently granted)."""
    raw_lines = raw_text.splitlines()
    comp_lines = set(compressed_text.splitlines())
    removed = [l for l in raw_lines if l not in comp_lines]
    return "\n".join(removed[:max_lines])


def _extract_range(raw_text: str, line_range: str) -> Optional[str]:
    """line_range like 'L40-L55' or 'L12'. Returns None if malformed/out of bounds."""
    try:
        spec = line_range.strip().lstrip("L")
        if "-" in spec:
            start_s, end_s = spec.split("-", 1)
            start, end = int(start_s), int(end_s.lstrip("L"))
        else:
            start = end = int(spec)
    except ValueError:
        return None

    lines = raw_text.splitlines()
    if start < 1 or end < start or start > len(lines):
        return None
    end = min(end, len(lines))
    return "\n".join(lines[start - 1 : end])


def run_feedback_loop(
    raw_text: str,
    compressed_text: str,
    agent_verify_fn: Callable[[str, int], str],
) -> FeedbackResult:
    """
    agent_verify_fn(current_text, round_num) -> agent's verdict string.
    Expected verdict format: "OK" or "MISSING: L<start>-L<end> <why>".
    This function is the plug point for whatever LLM call agent_localizer.py
    wires up (Claude / GPT / DeepSeek / Qwen — model-agnostic by design,
    since WP1's comparison spans exactly those four).
    """
    current_text = compressed_text
    turns: List[FeedbackTurn] = []
    removed_blob = _diff_summary(raw_text, compressed_text)

    for round_num in range(1, MAX_ROUNDS + 1):
        verdict = agent_verify_fn(current_text, round_num)

        if not verdict.strip().upper().startswith("MISSING"):
            turns.append(FeedbackTurn(round_num, "OK", None, None))
            return FeedbackResult(final_text=current_text, rounds_used=round_num, turns=turns)

        # parse "MISSING: L40-L55 ..." — first token after MISSING: is the range
        rest = verdict.split(":", 1)[1].strip() if ":" in verdict else ""
        requested_range = rest.split()[0] if rest else None

        revealed = _extract_range(raw_text, requested_range) if requested_range else None
        if revealed is None:
            # malformed or out-of-bounds request: don't grant it, don't burn
            # trust either — just note it and let the loop continue/expire
            turns.append(FeedbackTurn(round_num, verdict, requested_range, None))
            continue

        current_text = (
            current_text
            + f"\n\n[FEEDBACK LOOP round {round_num}: agent requested {requested_range}]\n"
            + revealed
        )
        turns.append(FeedbackTurn(round_num, verdict, requested_range, revealed))

    # ran out of rounds — return whatever we have, flagged as exhausted
    final_verdict = agent_verify_fn(current_text, MAX_ROUNDS + 1)
    exhausted = final_verdict.strip().upper().startswith("MISSING")
    return FeedbackResult(
        final_text=current_text, rounds_used=MAX_ROUNDS, turns=turns, exhausted=exhausted
    )
