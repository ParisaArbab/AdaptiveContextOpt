"""Conservative, *untrained* evidence-acquisition fallback for real bugs.

This rule is deliberately simpler than the proposed trained AITECH
marginal-harm controller. It uses ONLY pre-gold failure provenance:
- a production class appeared in a real stack trace; or
- a test's class name strongly suggests a production class.

If an already included candidate matches such a strong source clue,
the fallback **does not expand to L1** without independent learned
action-value evidence. Otherwise, it requests a feasible L1 detail
upgrade for the highest-ranked available candidate.

This is a falsifiable exploration policy, NOT a calibrated probability
or a validated root-cause predictor. In particular, names and stack
frames may be misleading. Evaluate against an always-upgrade arm on
untouched bugs before claiming improved accuracy/cost.
"""
from __future__ import annotations

from dataclasses import dataclass

from .failure_grounded_rerank import (
    FRAME, failing_test_class_hints, rerank, shortlist
)


@dataclass(frozen=True)
class GuardDecision:
    action: str
    reason: str
    target_method: str | None
    traced_candidate_count: int
    test_name_candidate_count: int


def decide_pre_gold(snapshot: dict, *, shortlist_size: int = 8) -> GuardDecision:
    """Return STOP or UPGRADE_L1 with NO gold labels and NO ML claims."""
    selected = shortlist(snapshot, rerank(snapshot), count=shortlist_size)
    frames = {
        c.replace("$", ".") for c, _ in
        FRAME.findall(snapshot["context"])
    }
    test_hints = failing_test_class_hints(snapshot["context"])
    trace_matches = []
    test_matches = []
    for candidate in selected["candidates"]:
        qualified = candidate["function_id"].split("(", 1)[0]
        owner = qualified.rsplit(".", 1)[0]
        short = owner.rsplit(".", 1)[-1]
        if owner in frames:
            trace_matches.append(candidate["function_id"])
        if short in test_hints:
            test_matches.append(candidate["function_id"])

    if trace_matches:
        return GuardDecision(
            "STOP", "selected_production_trace_class", None,
            len(trace_matches), len(test_matches))
    if test_matches:
        return GuardDecision(
            "STOP", "selected_test_name_class_clue", None,
            len(trace_matches), len(test_matches))
    return GuardDecision(
        "UPGRADE_L1", "no_source_identity_clue_in_shortlist",
        selected["candidates"][0]["function_id"], 0, 0)
