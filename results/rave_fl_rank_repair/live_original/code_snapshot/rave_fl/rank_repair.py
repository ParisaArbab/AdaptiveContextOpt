"""L1-first, budgeted rank repair with a frozen final ranker.

No training labels, patch, gold function or unobserved full answer is accepted.
The utility is a transparent ranking proxy, not a calibrated expected quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from rave_fl.adaptive import AdaptivePilot, fused_score
from rave_fl.core import Action
from rave_fl.pilot_types import Decision


REPAIR_MODES = frozenset({"repair_top", "repair_disagreement", "repair_suspicion", "repair_utility"})


@dataclass(frozen=True)
class RepairConfig:
    max_full: int = 4
    top_k: int = 5

    def __post_init__(self):
        if self.max_full < 0 or self.top_k < 1:
            raise ValueError("Invalid repair limit or ranking horizon")


def discounted_belief(order, beliefs, top_k):
    return sum(beliefs[ident] / rank for rank, ident in enumerate(order[:top_k], 1))


class RankRepairPilot(AdaptivePilot):
    def __init__(self, candidates, backend, config, repair_config=None):
        super().__init__(candidates, backend, config)
        if config.mode not in REPAIR_MODES:
            raise ValueError("Rank repair requires a repair_* policy")
        self.repair_config = repair_config or RepairConfig()
        self.repair_diagnostics = []
        self.initial_l1 = None

    def needs_more(self, candidate):
        # Unverified means no full-code confirmation, not deletion or a claim
        # that every such candidate needs another paid inspection.
        return self.status[candidate.function_id].level < 2

    def admitted(self, candidates, action, level, reason):
        decision = super().admitted(candidates, action, level, reason)
        return replace(decision, risks={}) if decision else None

    def ranking_proxy(self, candidate, estimate):
        scores = {c.function_id: fused_score(c, self.status[c.function_id]) for c in self.candidates}
        order = [c.function_id for c in self.ranked()]
        beliefs = {}
        for c in self.candidates:
            s = self.status[c.function_id]
            obs = s.observation
            prior = .10 + .50 * c.prior
            if obs is None or not obs.valid:
                beliefs[c.function_id] = prior
            elif s.level == 2:
                beliefs[c.function_id] = obs.suspicion
            else:
                beliefs[c.function_id] = (1 - obs.uncertainty) * obs.suspicion + obs.uncertainty * prior
        obs = self.status[candidate.function_id].observation
        if obs is None or not obs.valid:
            # Invalid cheap evidence has no trustworthy utility prediction.
            return {"gain_proxy": 0.0, "utility_per_token": 0.0, "invalid_l1": True}
        hypothetical = dict(scores)
        hypothetical[candidate.function_id] = obs.suspicion
        possible_order = sorted(self.candidates, key=lambda c: (-hypothetical[c.function_id], c.rank))
        base = discounted_belief(order, beliefs, self.repair_config.top_k)
        projected = discounted_belief([c.function_id for c in possible_order], beliefs, self.repair_config.top_k)
        gain = projected - base
        return {"gain_proxy": gain, "utility_per_token": max(0, gain) / estimate,
                "predicted_full_score_proxy": obs.suspicion,
                "current_rank": order.index(candidate.function_id) + 1,
                "projected_rank": [c.function_id for c in possible_order].index(candidate.function_id) + 1}

    def decide(self):
        if self.remaining <= 0:
            return Decision(Action.STOP, reason="budget_exhausted")
        pending = [c for c in self.candidates if self.status[c.function_id].level < 0]
        if pending:
            # Same candidate order, group composition and prompt as fixed_batch.
            # If a full batch cannot fit, do not silently substitute a cheaper
            # differently composed batch; stop with incomplete pool coverage.
            group = pending[:self.config.batch_size]
            decision = self.admitted(group, Action.BATCH_SCREEN, 1, "shared_l1_discovery")
            return decision or Decision(Action.STOP, reason="budget_admission_before_l1_coverage")
        if self.initial_l1 is None:
            self.initial_l1 = {"ranked_functions": [c.function_id for c in self.ranked()],
                               "tokens_used": self.spent,
                               "scores": {c.function_id: fused_score(c, self.status[c.function_id]) for c in self.candidates}}
        if sum(s.level == 2 for s in self.status.values()) >= self.repair_config.max_full:
            return Decision(Action.STOP, reason="full_repair_limit")
        rank = {c.function_id: i for i, c in enumerate(self.ranked(), 1)}
        options, diagnostics = [], []
        for c in self.candidates:
            s = self.status[c.function_id]
            if s.level >= 2:
                continue
            decision = self.admitted([c], Action.FULL_VERIFY, 2, self.config.mode)
            if decision is None:
                continue
            proxy = self.ranking_proxy(c, decision.estimated_tokens)
            if s.observation is None or not s.observation.valid:
                # Paid invalid L1 is recoverable; prioritize it without treating
                # it as a negative verdict or fabricating a model confidence.
                priority = (1, 0, -rank[c.function_id])
            elif self.config.mode == "repair_top":
                priority = (0, -rank[c.function_id], 0)
            elif self.config.mode == "repair_disagreement":
                priority = (0, c.features["ranker_disagreement"], -rank[c.function_id])
            elif self.config.mode == "repair_suspicion":
                priority = (0, s.observation.suspicion, -rank[c.function_id])
            else:
                priority = (0, proxy["utility_per_token"], -rank[c.function_id])
            diagnostics.append({"function_id": c.function_id, "estimate": decision.estimated_tokens,
                                "priority": list(priority), **proxy})
            # Numerical zero is not evidence of a safe stop. It is only the
            # explicit stopping rule of this untrained utility heuristic.
            if self.config.mode == "repair_utility" and priority[0] == 0 and priority[1] <= 1e-12:
                continue
            options.append((priority, decision))
        self.repair_diagnostics.append({"spent_tokens": self.spent, "remaining_tokens": self.remaining,
                                        "options": diagnostics})
        if options:
            return max(options, key=lambda item: item[0])[1]
        reason = "no_positive_rank_proxy" if diagnostics else "no_affordable_full_repair"
        return Decision(Action.STOP, reason=reason)

    def execute(self):
        result = super().execute()
        result.update({"risk_model": "not_used_untrained_rank_repair", "risk_calibrated": False,
                       "repair_config": asdict(self.repair_config), "initial_l1": self.initial_l1,
                       "repair_diagnostics": self.repair_diagnostics,
                       "utility_is_calibrated": False})
        return result
