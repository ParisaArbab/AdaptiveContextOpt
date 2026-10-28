"""Rank-risk interface. The bootstrap score is explicitly UNCALIBRATED."""
from __future__ import annotations

import json
import math
from pathlib import Path
from rave_fl.code_sketch import FEATURE_NAMES


class BootstrapRisk:
    name = "bootstrap_heuristic_uncalibrated"
    calibrated = False
    training_bugs: tuple[str, ...] = ()
    threshold = None

    def predict(self, f):
        # Estimate evidence loss, not probability that this method is the bug.
        return min(1.0, .08 + .30 * f["uncertainty"] + .22 * f["failure_relevance"]
                   + .15 * f["ranker_disagreement"] + .15 * (1 - f["compression_ratio"])
                   + .12 * min(1, f["complexity"] / 10)
                   + .20 * f["parse_error"] + .12 * f["sketch_truncated"])


class LearnedRisk:
    name = "lightgbm_isotonic_candidate_harm"
    calibrated = True

    def __init__(self, path: Path, evaluation_bug: str | None = None):
        import lightgbm as lgb
        data = json.loads(path.read_text())
        if data.get("schema") != 1 or data.get("features") != list(FEATURE_NAMES):
            raise ValueError("Risk artifact feature schema mismatch")
        self.training_bugs = tuple(data["train_bugs"] + data["calibration_bugs"])
        if evaluation_bug in self.training_bugs:
            raise ValueError("Leakage: evaluation bug occurred in training or calibration")
        self.model = lgb.Booster(model_str=data["booster"])
        self.x, self.y = data["calibration_x"], data["calibration_y"]
        self.threshold = float(data["risk_threshold"])
        self.metadata = {k: v for k, v in data.items() if k not in {"booster", "calibration_x", "calibration_y"}}
        if not 0 <= self.threshold <= 1 or not self.x or len(self.x) != len(self.y):
            raise ValueError("Invalid risk calibration artifact")
        if not all(math.isfinite(v) for v in self.x + self.y):
            raise ValueError("Nonfinite calibration values")

    def predict(self, f):
        import numpy as np
        values = np.asarray([[f[k] for k in FEATURE_NAMES]], dtype=float)
        score = float(self.model.predict(values, num_threads=1)[0])
        return float(np.interp(score, self.x, self.y))
