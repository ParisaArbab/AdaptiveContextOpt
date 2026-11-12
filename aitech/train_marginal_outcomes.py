"""Train/calibrate a three-outcome marginal-action model from real LLM pairs.

Only consumes independently frozen paired prediction outcomes:
  -1 harmful, 0 unchanged, +1 beneficial.
Model: multinomial logistic regression, project-disjoint train/calibration/test.
A scalar temperature is chosen using *calibration* projects only.
The held-out test is evaluated once. No training rows are bundled here.

This is an experiment pipeline, not an already trained controller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path
import json

from .marginal_action_policy import (
    ActionEstimate, EvidenceAction, FEATURE_NAMES, action_feature_vector
)
from .paired_action_outcomes import validate_pairs

CLASSES = (-1, 0, 1)


def _softmax(values: list[float]) -> list[float]:
    pivot = max(values)
    exps = [exp(x - pivot) for x in values]
    return [x / sum(exps) for x in exps]


@dataclass
class LearnedMarginalEstimator:
    """Pure-Python inference from verified project-disjoint calibration artifact."""
    artifact: dict

    @property
    def calibrated_on_independent_projects(self) -> bool:
        return bool(self.artifact.get("independent_project_calibration_verified", False))

    def predict(self, action: EvidenceAction) -> ActionEstimate:
        if not self.calibrated_on_independent_projects:
            raise ValueError("uncalibrated artifact cannot be used for decisions")
        values = action_feature_vector(action)
        if self.artifact.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("feature version mismatch")
        mu, std = self.artifact["mean"], self.artifact["scale"]
        normalized = [(a - b) / c for a, b, c in zip(values, mu, std)]
        logits = [sum(a*b for a, b in zip(weight, normalized)) + bias
                  for weight, bias in zip(self.artifact["coef"],
                                          self.artifact["intercept"])]
        probabilities = _softmax([value / self.artifact["temperature"] for value in logits])
        if self.artifact["classes"] != list(CLASSES) or len(probabilities) != 3:
            raise ValueError("model classes must be harm/neutral/improvement")
        return ActionEstimate(action.action_id,
                              p_improve=probabilities[2], p_harm=probabilities[0])


def _logloss(y: list[int], probs: list[list[float]]) -> float:
    return sum(-log(max(1e-12, p[CLASSES.index(v)]))
               for v, p in zip(y, probs)) / len(y)


def _brier(y: list[int], probs: list[list[float]]) -> float:
    return sum(sum((p[i] - float(v == CLASSES[i])) ** 2
                   for i in range(3)) for v, p in zip(y, probs)) / len(y)


def train(rows: list[dict], min_calibration_rows: int = 20) -> dict:
    """Fit only when independent projects and enough real outcome rows exist."""
    summary = validate_pairs(rows)
    parts = {k: [r for r in rows if r["split"] == k]
             for k in ("train", "calibration", "test")}
    if any(not v for v in parts.values()):
        raise ValueError("need nonempty independent train/calibration/test projects")
    if len({r["project"] for r in parts["train"]}) < 2:
        raise ValueError("need at least two training projects")
    if len(parts["calibration"]) < min_calibration_rows:
        raise ValueError("insufficient independent calibration observations")
    if set(r["outcome"] for r in parts["train"]) != set(CLASSES):
        raise ValueError("training needs harmful, neutral and improved action outcomes")
    if set(r["outcome"] for r in parts["calibration"]) != set(CLASSES):
        raise ValueError("calibration needs harmful, neutral and improved outcomes")

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def xy(part):
        return (
            np.asarray([action_feature_vector(EvidenceAction(**r["action"]))
                        for r in part], dtype=float),
            np.asarray([r["outcome"] for r in part], dtype=int)
        )

    x_train, y_train = xy(parts["train"])
    x_cal, y_cal = xy(parts["calibration"])
    x_test, y_test = xy(parts["test"])
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=0.5, max_iter=600, random_state=42)
    model.fit(scaler.transform(x_train), y_train)
    if list(model.classes_) != list(CLASSES):
        raise AssertionError("missing outcome class")

    cal_logits = model.decision_function(scaler.transform(x_cal))
    test_logits = model.decision_function(scaler.transform(x_test))
    if cal_logits.ndim != 2 or cal_logits.shape[1] != 3:
        raise AssertionError("expected three outcome logits")

    # Fixed candidate grid selected exclusively on calibration projects.
    temperatures = (0.5, 0.7, 1.0, 1.3, 1.8, 2.5, 3.5, 5.0)
    def probs(logits, temp):
        return [_softmax([float(value) / temp for value in row])
                for row in logits]
    optimum = min(temperatures, key=lambda t: _logloss(
        y_cal.tolist(), probs(cal_logits, t)))
    calibration_probs = probs(cal_logits, optimum)
    heldout_probs = probs(test_logits, optimum)
    artifact = {
        "schema": 1,
        "estimator": "multinomial_logistic_with_scalar_temperature",
        "classes": list(CLASSES),
        "feature_names": list(FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "temperature": optimum,
        "independent_project_calibration_verified": True,
        "source_pairs_sha256": summary["sha256"],
        "project_split": summary["project_split"],
        "model_id": summary["model_id"],
        "outcome_definition": summary["outcome_definition"],
        "common_input_token_ceiling": summary["common_input_token_ceiling"],
        "counts": {key: len(value) for key, value in parts.items()},
        "calibration_metrics": {
            "multiclass_logloss": _logloss(y_cal.tolist(), calibration_probs),
            "multiclass_brier": _brier(y_cal.tolist(), calibration_probs),
        },
        "heldout_test_metrics": {
            "multiclass_logloss": _logloss(y_test.tolist(), heldout_probs),
            "multiclass_brier": _brier(y_test.tolist(), heldout_probs),
        },
        "limitations": [
            "Calibration is empirical, not a statistical risk guarantee",
            "Training requires genuine paired LLM outputs, not synthetic labels",
            "Held-out data must remain separate from hyperparameter selection",
            "Prompt ceiling is equal but realized prompt token counts can differ",
        ],
    }
    return artifact


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.pairs_jsonl.read_text().splitlines()
            if line.strip()]
    artifact = train(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print("Saved calibrated experiment artifact:", args.output)


if __name__ == "__main__":
    main()
