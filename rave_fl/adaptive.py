"""Closed-loop allocation: policy -> paid observation -> state -> policy.

The API accepts no gold labels. All arms share the same evidence-fusion ranker.
The legacy implementation remains available for reproducing the October report.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from rave_fl.code_sketch import state_features
from rave_fl.core import Action
from rave_fl.pilot_types import CandidateState, Decision, PolicyConfig
from rave_fl.risk import BootstrapRisk


def fused_score(candidate, state):
    """Retain a prior for every method; being inspected is never a sort key.

    Pilot scores are ranking scores, NOT calibrated fault probabilities.
    Complete-code evidence uses its suspicion directly, matching the full arm.
    """
    prior = .10 + .50 * candidate.prior
    obs = state.observation
    if obs is None or not obs.valid:
        return prior
    if state.level == 2:
        return obs.suspicion
    weight = {0: .30, 1: .65}[state.level] * (1 - obs.uncertainty)
    return (1 - weight) * prior + weight * obs.suspicion


class AdaptivePilot:
    def __init__(self, candidates, backend, config: PolicyConfig | None = None, risk=None):
        self.candidates = list(candidates)
        if not self.candidates or len({c.function_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("A nonempty unique candidate pool is required")
        if any(not math.isfinite(c.prior) or not 0 <= c.prior <= 1 or not c.source.strip()
               for c in self.candidates):
            raise ValueError("Each candidate needs real source and a prior in [0,1]")
        self.by_id = {c.function_id: c for c in self.candidates}
        self.backend, self.config, self.risk = backend, config or PolicyConfig(), risk or BootstrapRisk()
        self.status = {c.function_id: CandidateState() for c in self.candidates}
        self.spent = self.input_tokens = self.output_tokens = 0
        self.calls, self.trace, self.snapshots = [], [], []
        self.duration = 0.0
        self.fusion = None
        self.cost_aware = True

    @property
    def remaining(self):
        # Negative actual balance is preserved in output; never conceal overshoot.
        return self.config.token_budget - self.spent

    def feature(self, c):
        return state_features(c, self.status[c.function_id],
                              max(0, self.remaining) / max(1, self.config.token_budget),
                              sum(s.level >= 0 for s in self.status.values()) / len(self.status))

    def risk_score(self, c):
        value = float(self.risk.predict(self.feature(c)))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Risk model must return a finite value in [0,1]")
        return value

    def scores(self):
        if self.fusion is not None:
            return self.fusion.scores(self.candidates, self.status)
        return {c.function_id: fused_score(c, self.status[c.function_id]) for c in self.candidates}

    def ordered(self, scores):
        if self.fusion is not None:
            return self.fusion.order(self.candidates, scores)
        return sorted(self.candidates, key=lambda c: (-scores[c.function_id], c.rank))

    def ranked(self):
        return self.ordered(self.scores())

    def needs_more(self, c):
        s, cfg = self.status[c.function_id], self.config
        if s.level >= 2:
            return False
        if s.observation is None or not s.observation.valid:
            return True
        obs = s.observation
        confidence_need = obs.uncertainty >= cfg.uncertainty_threshold or obs.suspicion >= cfg.suspicion_threshold
        if cfg.mode == "confidence":
            return confidence_need
        threshold = self.risk.threshold if self.risk.threshold is not None else cfg.risk_threshold
        return (confidence_need or self.risk_score(c) >= threshold
                or c.features["parse_error"] > 0 or bool(obs.missing_evidence))

    def admitted(self, candidates, action, level, reason):
        estimate = self.backend.estimate(candidates, action, level)
        if not isinstance(estimate, int) or estimate < 1:
            raise ValueError("Invalid backend token estimate")
        if estimate > self.remaining:
            return None
        if estimate > getattr(self.backend, "context_window", math.inf):
            return None
        return Decision(action, tuple(c.function_id for c in candidates), level, reason,
                        estimate, {c.function_id: self.risk_score(c) for c in candidates})

    def decide(self):
        cfg = self.config
        if self.remaining <= 0:
            return Decision(Action.STOP, reason="budget_exhausted")
        if cfg.mode in {"full", "rank_threshold"}:
            pending = [c for c in self.candidates if self.status[c.function_id].level < 2
                       and (cfg.mode == "full" or c.prior >= .60)]
            for c in pending:
                d = self.admitted([c], Action.FULL_VERIFY, 2, "full_baseline" if cfg.mode == "full" else "legacy_rank_gate")
                if d:
                    return d
            return Decision(Action.STOP, reason="budget_admission" if pending else "policy_complete")
        uninspected = [c for c in self.candidates if self.status[c.function_id].level < 0]
        if uninspected:
            # Screen the entire supplied pool before early stopping or deep checks.
            # This prevents the old .60 gate from permanently hiding the tail.
            level = 1 if cfg.mode in {"fixed_batch", "fixed_sketch"} else 0
            size = 1 if cfg.mode == "fixed_sketch" else cfg.batch_size
            for offset in range(0, len(uninspected), size):
                group = uninspected[offset:offset + size]
                for n in range(len(group), 0, -1):
                    d = self.admitted(group[:n], Action.BATCH_SCREEN, level, "pool_discovery")
                    if d:
                        return d
            return Decision(Action.STOP, reason="budget_admission_before_pool_coverage")
        if cfg.mode in {"fixed_batch", "fixed_sketch", "probe_l0"}:
            return Decision(Action.STOP, reason="fixed_screen_complete")
        pending = [c for c in self.candidates if self.needs_more(c)]
        if not pending:
            return Decision(Action.STOP, reason="all_required_checks_complete")
        ranking = self.ranked()
        gap = (self.scores()[ranking[0].function_id] -
               self.scores()[ranking[1].function_id]) if len(ranking) > 1 else 1
        threshold = self.risk.threshold if self.risk.threshold is not None else cfg.risk_threshold
        # Learned candidate harm is not a calibrated issue-level stop model.
        # Require full confirmation and no invalid/uncertain/high-risk pending item.
        if (cfg.early_stop and self.risk.calibrated and gap >= cfg.stop_gap
                and self.status[ranking[0].function_id].level == 2
                and all(self.status[c.function_id].observation.valid
                        and self.status[c.function_id].observation.uncertainty < cfg.uncertainty_threshold
                        and self.risk_score(c) < threshold for c in pending)):
            return Decision(Action.STOP, reason="conservative_calibrated_risk_stop")
        choices = []
        for c in pending:
            s = self.status[c.function_id]
            # Parse loss or invalid feedback must not be trusted at another cheap level.
            direct = c.features["parse_error"] or (s.observation and not s.observation.valid)
            level = 2 if s.level >= 1 or direct else 1
            decision_reason = "feedback_escalation"
            if cfg.mode == "adaptive" and self.cost_aware and level == 1:
                expanded_cost = self.backend.estimate([c], Action.EXPAND_CONTEXT, 1)
                full_cost = self.backend.estimate([c], Action.FULL_VERIFY, 2)
                # Avoid buying a sketch that is already as expensive as full
                # code, or is likely to need a second paid full-code inspection.
                # With bootstrap risk this is a cost proxy, not a probability.
                risk_weighted_cost = expanded_cost + self.risk_score(c) * full_cost
                if (risk_weighted_cost >= full_cost and full_cost <= self.remaining
                        and full_cost <= getattr(self.backend, "context_window", math.inf)):
                    level = 2
                    decision_reason = "skip_uneconomic_expansion"
            action = Action.FULL_VERIFY if level == 2 else Action.EXPAND_CONTEXT
            d = self.admitted([c], action, level, decision_reason)
            if d is None:
                continue
            f = c.features
            obs = s.observation
            uncertainty = obs.uncertainty if obs else 1
            suspicion = obs.suspicion if obs else .5
            # Value per estimated token adapts as budget, risk and observations change.
            if cfg.mode == "confidence":
                value = .10 + uncertainty + suspicion
            else:
                value = (.10 + self.risk_score(c) + .7 * uncertainty + .6 * suspicion
                         + .5 * f["failure_relevance"] + .3 * f["ranker_disagreement"])
            choices.append((value / d.estimated_tokens, -c.rank, d))
        if choices:
            return max(choices, key=lambda x: (x[0], x[1]))[2]
        return Decision(Action.STOP, reason="budget_admission_with_unresolved_candidates")

    def execute(self):
        reason = "max_steps"
        for step in range(self.config.max_steps):
            d = self.decide()
            self.trace.append({"step": step, **asdict(d), "action": d.action.value,
                               "spent_before": self.spent, "remaining_before": self.remaining,
                               "inspected_count": sum(s.level >= 0 for s in self.status.values())})
            if d.action == Action.STOP:
                reason = d.reason
                break
            selected = [self.by_id[i] for i in d.ids]
            result = self.backend.execute(selected, d.action, d.level)
            # Always charge the entire call exactly once, including invalid JSON.
            self.spent += result.total_tokens
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self.duration += result.duration_seconds
            ids = [o.function_id for o in result.observations]
            if len(ids) != len(d.ids) or set(ids) != set(d.ids):
                raise ValueError("Backend violated the one-observation-per-ID contract after a paid call")
            self.calls.append({"action": d.action.value, "level": d.level, "candidate_ids": list(d.ids),
                               **asdict(result)})
            for obs in result.observations:
                c, s = self.by_id[obs.function_id], self.status[obs.function_id]
                if d.level <= s.level:
                    raise ValueError("Evidence levels must strictly increase")
                s.level, s.observation = d.level, obs
                s.history.append({"action": d.action.value, "level": d.level, **asdict(obs)})
            # Capture only inference-visible state, for later offline risk labels.
            for c in selected:
                self.snapshots.append({"function_id": c.function_id, "level": d.level,
                                       "features": self.feature(c),
                                       "observation": asdict(self.status[c.function_id].observation)})
            self.trace[-1].update(actual_tokens=result.total_tokens, spent_after=self.spent,
                                  remaining_after=self.remaining, response_error=result.error)
        else:
            self.trace.append({"action": Action.STOP.value, "reason": reason,
                               "spent_before": self.spent, "remaining_before": self.remaining})
        ranking = self.ranked()
        return {
            "mode": self.config.mode, "config": asdict(self.config),
            "risk_model": self.risk.name, "risk_calibrated": self.risk.calibrated,
            "ranker": self.fusion.name if self.fusion is not None else "shared_prior_evidence_fusion_v1", "ranked_functions": [c.function_id for c in ranking],
            "ranking_scores": self.scores(),
            "actual_input_tokens": self.input_tokens, "actual_output_tokens": self.output_tokens,
            "actual_tokens": self.spent, "tokens_remaining": self.remaining,
            "budget_exceeded": self.remaining < 0, "budget_is_estimated_admission": True,
            "llm_duration_seconds": self.duration, "call_count": len(self.calls),
            "screened_count": sum(s.level >= 0 for s in self.status.values()),
            "full_verified_count": sum(s.level == 2 for s in self.status.values()),
            "stop_reason": reason, "trace": self.trace, "calls": self.calls,
            "states": {i: asdict(s) for i, s in self.status.items()}, "risk_snapshots": self.snapshots,
            "unresolved_ids": [c.function_id for c in self.candidates if self.needs_more(c)],
            "failed_full_ids": [i for i, s in self.status.items() if s.level == 2 and not s.observation.valid],
        }
