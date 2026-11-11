"""Provenance-gated marginal action decisions for code debugging.

Research hypothesis: choose STOP, RETRIEVE, GRAPH, or UPGRADE when the
calibrated expected downstream benefit exceeds harm and true token/time
cost. This module separates a *policy interface* from an estimator.

CRITICAL: No probabilities are fabricated here. No offline heuristic is
called calibrated. Use a fitted, independently evaluated outcome model
for the probability interface, or the policy abstains by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol


ACTIONS = ("RETRIEVE", "GRAPH", "UPGRADE_L1", "UPGRADE_FULL")


@dataclass(frozen=True)
class EvidenceAction:
    action_id: str
    kind: str
    target_id: str
    incremental_input_tokens: int
    incremental_latency_ms: float
    source_revision: str
    provenance: str
    compiler_binding_verified: bool = False
    requires_call_binding: bool = False
    support_sufficiency_audited: bool = False

    def validate(self) -> None:
        if self.kind not in ACTIONS:
            raise ValueError("unsupported action")
        if not self.action_id or not self.target_id:
            raise ValueError("missing action identity")
        if self.incremental_input_tokens <= 0:
            raise ValueError("incremental token cost must be positive")
        if not isfinite(self.incremental_latency_ms) or self.incremental_latency_ms < 0:
            raise ValueError("invalid latency estimate")
        if not self.source_revision.strip() or not self.provenance.strip():
            raise ValueError("action requires reproducible provenance")


@dataclass(frozen=True)
class ActionEstimate:
    action_id: str
    p_improve: float
    p_harm: float
    gain_if_improved: float = 1.0
    loss_if_harmed: float = 1.0

    def validate(self) -> None:
        vals = (self.p_improve, self.p_harm,
                self.gain_if_improved, self.loss_if_harmed)
        if any(not isfinite(v) for v in vals):
            raise ValueError("non-finite estimate")
        if not 0 <= self.p_improve <= 1 or not 0 <= self.p_harm <= 1:
            raise ValueError("invalid probabilities")
        if self.p_improve + self.p_harm > 1 + 1e-9:
            raise ValueError("gain and harm outcomes must be exclusive")
        if self.gain_if_improved < 0 or self.loss_if_harmed < 0:
            raise ValueError("negative outcome magnitude")


class CalibratedEstimator(Protocol):
    calibrated_on_independent_projects: bool

    def predict(self, action: EvidenceAction) -> ActionEstimate: ...


@dataclass(frozen=True)
class ActionDecision:
    action_id: str | None
    decision: str
    expected_net_value: float
    eligible_action_count: int
    rejection_reasons: tuple[tuple[str, str], ...]


def decide_next_action(
    actions: list[EvidenceAction],
    estimator: CalibratedEstimator | None,
    remaining_input_tokens: int,
    *,
    token_penalty: float = 0.001,
    latency_penalty_per_ms: float = 0.000001,
) -> ActionDecision:
    """Choose the best permitted action or STOP, without using patch labels.

    Requires an estimator specifically calibrated on *independent projects*;
    otherwise abstains. A verified call binding proves identity but does
    NOT prove that the support path is causally sufficient.

    Estimators are an interface, not proof their training data were valid.
    That must be audited with a separate outcome-data contract.
    """
    if remaining_input_tokens < 0:
        raise ValueError("negative remaining budget")
    if any(not isfinite(x) or x < 0 for x in
           (token_penalty, latency_penalty_per_ms)):
        raise ValueError("invalid cost coefficients")
    ids = [a.action_id for a in actions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate action IDs")
    for action in actions:
        action.validate()

    rejections: list[tuple[str, str]] = []
    eligible: list[tuple[float, str]] = []
    if estimator is None or not estimator.calibrated_on_independent_projects:
        return ActionDecision(
            None, "STOP_UNCALIBRATED", 0.0, 0,
            tuple((a.action_id, "no_independent_calibration") for a in actions)
        )

    for action in actions:
        if action.incremental_input_tokens > remaining_input_tokens:
            rejections.append((action.action_id, "token_budget_exceeded"))
            continue
        if action.requires_call_binding and not action.compiler_binding_verified:
            rejections.append((action.action_id, "unverified_call_binding"))
            continue
        # A sufficiently grounded binding may still be optional; audited
        # sufficiency only gates claims about dependency *completeness*,
        # not whether a verified call is worth retrieving.
        prediction = estimator.predict(action)
        prediction.validate()
        if prediction.action_id != action.action_id:
            raise ValueError("estimator returned action for wrong identifier")
        value = (
            prediction.p_improve * prediction.gain_if_improved
            - prediction.p_harm * prediction.loss_if_harmed
            - token_penalty * action.incremental_input_tokens
            - latency_penalty_per_ms * action.incremental_latency_ms
        )
        eligible.append((value, action.action_id))

    if not eligible:
        return ActionDecision(None, "STOP_NO_ELIGIBLE_ACTIONS", 0.0, 0,
                              tuple(rejections))
    best_value, best_id = sorted(eligible, key=lambda pair: (-pair[0], pair[1]))[0]
    if best_value <= 0:
        return ActionDecision(None, "STOP_NONPOSITIVE_VALUE", 0.0, len(eligible),
                              tuple(rejections))
    return ActionDecision(best_id, "ACQUIRE", round(best_value, 12),
                          len(eligible), tuple(rejections))
