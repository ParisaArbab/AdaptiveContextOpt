"""Conservative MODEL-FREE STOP gate for candidate *class* localization.

Novelty hypothesis (not established): a controller may save LLM tokens
when source-backed failure evidence sufficiently identifies a class.

This module is a transparent, heuristic pre-LLM gate, NOT a trained or
calibrated predictor of marginal action value. It returns a possible
class hint only when a failing test naming convention unambiguously
matches a ranked production class. A stronger LLM or another tool
must handle all other states. Never claim method or line-level fault
localization, causal explanation, or guaranteed safe abstention.
"""
from __future__ import annotations

from dataclasses import dataclass
from .test_name_anchored_reranker import rank_with_test_hints


@dataclass(frozen=True)
class GateResult:
    action: str
    candidate_method: str | None
    predicted_class: str | None
    reason: str
    model_input_tokens: int
    model_output_tokens: int


def decide_model_free_stop(snapshot: dict, *, require_exact_test_match: bool = True
                           ) -> GateResult:
    if snapshot.get("provenance", {}).get("gold_patches_loaded") is not False:
        raise ValueError("source snapshot is not explicitly pre-gold")
    ranked = rank_with_test_hints(snapshot)
    if not ranked:
        return GateResult("ESCALATE", None, None, "no_candidates", 0, 0)
    winner = ranked[0]
    owner = winner["method"].split("(", 1)[0].rsplit(".", 1)[0]
    if not winner["test_name_hint"]:
        return GateResult("ESCALATE", None, None, "no_test_class_evidence", 0, 0)
    if require_exact_test_match and winner["test_name_boost"] != 30.0:
        return GateResult("ESCALATE", None, None, "ambiguous_test_class_mapping", 0, 0)
    # This STOP is justified only as a *class-level heuristic*.
    # It must be evaluated on untouched projects before promotion.
    return GateResult("STOP_CLASS_HINT", winner["method"], owner,
                      "conventional_test_class_exact_source_owner_match",
                      0, 0)
