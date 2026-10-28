"""Frozen historical fusion plus explicit gold-free score and tie ablations."""
from dataclasses import replace
import hashlib
from rave_fl.adaptive import fused_score
from rave_fl.pilot_types import CandidateState

FUSIONS = ("shared_prior_evidence_fusion_v1", "consistent_score", "rrf", "prior_aware")


class Fusion:
    def __init__(self, name=FUSIONS[0], tie="original", rounding=None, rrf_k=60):
        if name not in FUSIONS or tie not in {"original", "identifier", "hash"}:
            raise ValueError("Unknown fusion or tie strategy")
        if rounding is not None and not 0 <= rounding <= 12:
            raise ValueError("Rounding must be 0..12 or unset")
        if rrf_k < 1:
            raise ValueError("RRF k must be positive")
        self.name, self.tie, self.rounding, self.rrf_k = name, tie, rounding, rrf_k

    def scores(self, candidates, states):
        result = {}
        observations = [
            s.observation.suspicion
            for s in states.values()
            if s.observation and s.observation.valid
        ]
        for c in candidates:
            s = states[c.function_id]
            o = s.observation
            prior = 0.1 + 0.5 * c.prior
            if self.name == FUSIONS[0]:
                value = fused_score(c, s)
            elif self.name == "rrf":
                value = 1 / (self.rrf_k + c.rank) + (
                    1 / (self.rrf_k + 1 + sum(v > o.suspicion for v in observations))
                    if o and o.valid
                    else 0
                )
            elif not o or not o.valid:
                value = prior
            elif self.name == "consistent_score":
                value = o.suspicion
            else:
                weight = 0.65 * (1 - o.uncertainty)
                value = (1 - weight) * prior + weight * o.suspicion
            result[c.function_id] = value
        return result

    def order(self, candidates, scores):
        def key(c):
            score = scores[c.function_id]
            score = round(score, self.rounding) if self.rounding is not None else score
            tie = (
                c.rank
                if self.tie == "original"
                else c.function_id
                if self.tie == "identifier"
                else hashlib.sha256(c.function_id.encode()).hexdigest()
            )
            return -score, tie

        return sorted(candidates, key=key)

    def project(self, candidates, states, candidate):
        hypothetical = dict(states)
        s = states[candidate.function_id]
        hypothetical[candidate.function_id] = CandidateState(2, s.observation)
        return self.scores(candidates, hypothetical)

    def config(self):
        return {
            "name": self.name,
            "tie": self.tie,
            "rounding": self.rounding,
            "rrf_k": self.rrf_k,
            "calibrated": False,
        }
