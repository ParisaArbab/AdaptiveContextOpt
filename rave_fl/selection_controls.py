"""Simple L1-first controls. Frozen V2 utility and ranker remain untouched.

Only current paid observations and input ranks enter priorities. Values below
are dimensionless ranking heuristics, never calibrated fault probabilities.
"""
from dataclasses import replace
import statistics

from rave_fl.adaptive import fused_score
from rave_fl.core import Action
from rave_fl.pilot_types import Decision, PolicyConfig
from rave_fl.rank_repair import RankRepairPilot, RepairConfig

POLICIES = ("fixed_l1", "highest_suspicion", "ranker_disagreement", "ranking_ambiguity", "v2_utility", "uncertainty_aware")
RANKERS = ("SBIR", "Ochiai", "BoostN")


def rank_percentiles(ranks, depths):
    """Top-list absence is right-censoring, encoded as worst percentile 1.

    Each list has its own truncation depth, so ranks on unequal scales can be
    compared. This measures observed disagreement, not unknown true rank.
    """
    if set(depths) != set(RANKERS) or any(v < 1 for v in depths.values()):
        raise ValueError("Require positive depth for all three rankers")
    if set(ranks) - set(RANKERS):
        raise ValueError("Unknown ranking signal")
    if any(not 1 <= r <= depths[m] for m, r in ranks.items()):
        raise ValueError("Rank outside recorded truncation depth")
    return [ranks[m] / (depths[m] + 1) if m in ranks else 1.0 for m in RANKERS]


def cheap_features(candidates, states, metadata, depths, fusion=None):
    scores = fusion.scores(candidates, states) if fusion is not None else {c.function_id: fused_score(c, states[c.function_id]) for c in candidates}
    order = fusion.order(candidates, scores) if fusion is not None else sorted(candidates, key=lambda c: (-scores[c.function_id], c.rank))
    gaps = [scores[a.function_id] - scores[b.function_id] for a, b in zip(order, order[1:])]
    scale = statistics.median([g for g in gaps if g > 1e-12]) if any(g > 1e-12 for g in gaps) else 1.0
    n = len(order)
    result = {}
    for pos, c in enumerate(order):
        ident, rank = c.function_id, pos + 1
        obs = states[ident].observation
        signal = rank_percentiles(metadata[ident]["ranks"], depths)
        adjacent = ([gaps[pos - 1]] if pos else []) + ([gaps[pos]] if pos < n - 1 else [])
        closeness = 1 / (1 + min(adjacent) / scale) if adjacent else 0.0
        movement = abs(c.rank - rank) / max(1, n - 1)
        dispersion = statistics.pstdev(signal + [rank / (n + 1)]) * 2  # std([0,1]) <= .5
        boundary = 1 / (1 + min(abs(rank - 1), abs(rank - min(5, n))))
        uncertainty = obs.uncertainty if obs else 1.0
        result[ident] = {"current_rank": rank, "current_score": scores[ident],
                         "rank_percentiles": signal, "disagreement": max(signal) - min(signal),
                         "adjacent_closeness": closeness, "rank_movement": movement,
                         "rank_dispersion": dispersion, "boundary_proximity": boundary,
                         "reported_uncertainty": uncertainty,
                         "raw_l1_suspicion": obs.suspicion if obs else .5,
                         "evidence_support": statistics.mean([c.features.get("failure_relevance", 0),
                                                              c.features.get("trace_match", 0)]),
                         "gap_above": gaps[pos-1] if pos else None,
                         "gap_below": gaps[pos] if pos < n-1 else None,
                         "ambiguity": statistics.mean([closeness, movement, dispersion, boundary, uncertainty])}
    return result


class SelectionControl(RankRepairPilot):
    def __init__(self, candidates, backend, config, repair_config, policy, metadata, depths):
        if policy not in POLICIES or policy == "v2_utility":
            raise ValueError("Use the unchanged RankRepairPilot for V2 utility")
        super().__init__(candidates, backend, replace(config, mode="repair_suspicion"),
                         RepairConfig(0, repair_config.top_k) if policy == "fixed_l1" else repair_config)
        self.policy, self.metadata, self.depths = policy, metadata, depths
        self.control_diagnostics = []

    def decide(self):
        if any(s.level < 0 for s in self.status.values()) or self.remaining <= 0:
            return super().decide()
        # Let the frozen implementation capture initial L1 and enforce the cap.
        if self.initial_l1 is None:
            self.initial_l1 = {"ranked_functions": [c.function_id for c in self.ranked()],
                               "tokens_used": self.spent,
                               "scores": self.scores()}
        if sum(s.level == 2 for s in self.status.values()) >= self.repair_config.max_full:
            return Decision(Action.STOP, reason="full_repair_limit")
        pending = [c for c in self.candidates if self.status[c.function_id].level < 2]
        best = self.ranked()[0]
        incumbent = self.status[best.function_id]
        if (incumbent.level == 2 and incumbent.observation.valid and
                all(self.status[c.function_id].observation and self.status[c.function_id].observation.valid and
                    incumbent.observation.suspicion > self.status[c.function_id].observation.suspicion for c in pending)):
            return Decision(Action.STOP, reason="full_incumbent_above_remaining_l1_heuristic")
        features = cheap_features(self.candidates, self.status, self.metadata, self.depths, self.fusion)
        highest = max(pending, key=lambda c: (self.status[c.function_id].observation.suspicion
                      if self.status[c.function_id].observation else .5,
                      -features[c.function_id]['current_rank'], -c.rank)).function_id
        options, rows = [], []
        for c in pending:
            decision = self.admitted([c], Action.FULL_VERIFY, 2, self.policy)
            obs, f = self.status[c.function_id].observation, features[c.function_id]
            if decision is None:
                rows.append({"function_id": c.function_id, "estimate": self.backend.estimate([c], Action.FULL_VERIFY, 2),
                             "eligible": False, "rejection": "budget_or_context_admission", **f})
                continue
            invalid = obs is None or not obs.valid
            score = (obs.suspicion if not invalid else 0) if self.policy == "highest_suspicion" else f[
                "disagreement" if self.policy == "ranker_disagreement" else "ambiguity"]
            if self.policy == "uncertainty_aware":
                # A deliberately simple development heuristic. No fitted weights,
                # future outcomes, observed action costs or labels enter it.
                value = (.4 * f["raw_l1_suspicion"] + .2 * f["disagreement"] +
                         .25 * f["ambiguity"] + .15 * (1 - f["evidence_support"]))
                score = value * (.5 + .5 * self.remaining / max(1, self.config.token_budget)) / decision.estimated_tokens
            worth = ((c.function_id == highest) if self.policy == "highest_suspicion" else
                     invalid or score > 1e-12)
            priority = (int(invalid), score, -f["current_rank"], -c.rank)
            rows.append({"function_id": c.function_id, "estimate": decision.estimated_tokens,
                         "selection_score": score, "eligible": worth, "priority": list(priority), **f})
            if worth:
                options.append((priority, decision))
        self.control_diagnostics.append({"spent_tokens": self.spent, "remaining_tokens": self.remaining, "options": rows})
        if options:
            return max(options, key=lambda item: item[0])[1]
        return Decision(Action.STOP, reason="no_positive_cheap_signal" if rows else "no_affordable_full_repair")

    def execute(self):
        result = super().execute()
        result.update(selection_policy=self.policy, control_diagnostics=self.control_diagnostics,
                      selection_is_calibrated=False)
        return result


def make_engine(candidates, backend, budget, cap, policy, metadata, depths, batch_size=4):
    config = PolicyConfig(token_budget=budget, batch_size=batch_size, mode="repair_utility")
    repair = RepairConfig(cap, 5)
    if policy == "v2_utility":
        return RankRepairPilot(candidates, backend, config, repair)
    return SelectionControl(candidates, backend, config, repair, policy, metadata, depths)
