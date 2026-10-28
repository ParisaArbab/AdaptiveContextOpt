"""Offline LightGBM training with disjoint bug-group calibration and test sets.

Usage: python -m rave_fl.train_risk --data risk.jsonl --split split.json --output risk_model.json
Split JSON: {"train":["Time-1",...],"calibration":[...],"test":[...]}.
Insufficient labels are an error, never a fabricated trained model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from rave_fl.code_sketch import FEATURE_NAMES


def check_split(rows, split):
    if set(split) != {"train", "calibration", "test"}:
        raise ValueError("Split must contain train, calibration and test bug lists")
    groups = [set(split[k]) for k in ("train", "calibration", "test")]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Train/calibration/test bug groups must be nonempty and disjoint")
    known = set.union(*groups)
    if any(r["bug"] not in known for r in rows):
        raise ValueError("Every data bug must have an explicit split")
    for row in rows:
        if row.get("label_definition") not in {"gold_false_negative_introduced_relative_to_full", "counterfactual_cheap_context_MRR_loss_relative_to_full_action"}:
            raise ValueError("Mixed or unknown risk label definitions")
        if set(row["features"]) != set(FEATURE_NAMES):
            raise ValueError("Unknown, missing, or oracle risk features")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row["features"].values()):
            raise ValueError("Risk features must be finite")
        if row["label"] not in (0, 1):
            raise ValueError("Labels must be binary")
    result = {k: [r for r in rows if r["bug"] in set(split[k])] for k in split}
    if len({r["label_definition"] for r in rows}) != 1:
        raise ValueError("Mixed risk label definitions")
    for key, values in result.items():
        if len({r["bug"] for r in values}) != len(set(split[key])):
            raise ValueError(f"Some {key} bugs have no observations")
        if {r["label"] for r in values} != {0, 1}:
            raise ValueError(f"{key} needs both harmful and safe examples; collect more independent bugs")
    return result


def metrics(y, p):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y, p = np.asarray(y), np.asarray(p)
    ece = 0.0
    for low in np.arange(0, 1, .1):
        mask = (p >= low) & ((p < low + .1) if low < .89 else (p <= 1))
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return {"AUROC": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ECE_10_bins": float(ece), "n": len(y), "positives": int(y.sum())}


def train(rows, split, max_cheap_harm=.05, min_admitted=10, target="candidate"):
    import numpy as np
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    if target not in {"candidate", "ranking"}:
        raise ValueError("Unknown risk target")
    if target == "ranking":
        if any("ranking_harm_label" not in r or r.get("ranking_label_definition") !=
               "counterfactual_cheap_context_MRR_loss_relative_to_full_action" for r in rows):
            raise ValueError("Ranking target requires offline counterfactual ranking labels")
        rows = [{**r, "label": r["ranking_harm_label"],
                 "label_definition": "counterfactual_cheap_context_MRR_loss_relative_to_full_action"} for r in rows]
    parts = check_split(rows, split)
    if not 0 <= max_cheap_harm <= 1 or min_admitted < 1:
        raise ValueError("Invalid calibration constraints")
    def xy(values):
        return np.asarray([[r["features"][k] for k in FEATURE_NAMES] for r in values]), np.asarray([r["label"] for r in values])
    tx, ty = xy(parts["train"])
    cx, cy = xy(parts["calibration"])
    vx, vy = xy(parts["test"])
    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=7, max_depth=3,
                               min_child_samples=max(2, min(10, len(ty) // 10)),
                               reg_lambda=1.0, random_state=0, n_jobs=1, verbosity=-1)
    model.fit(tx, ty, feature_name=list(FEATURE_NAMES))
    raw_c = model.booster_.predict(cx, num_threads=1)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw_c, cy)
    cp = iso.predict(raw_c)
    # Threshold chosen only on calibration bugs. No test outcome enters selection.
    threshold, admitted_count, observed_harm = 0.0, 0, None
    for t in sorted(set([float(v) for v in cp] + [1.0])):
        mask = cp < t
        n = int(mask.sum())
        if n >= min_admitted and float(cy[mask].mean()) <= max_cheap_harm:
            threshold, admitted_count, observed_harm = t, n, float(cy[mask].mean())
    vp = iso.predict(model.booster_.predict(vx, num_threads=1))
    raw_v = model.booster_.predict(vx, num_threads=1)
    return {"schema": 1, "features": list(FEATURE_NAMES), "booster": model.booster_.model_to_string(),
            "calibration_x": iso.X_thresholds_.tolist(), "calibration_y": iso.y_thresholds_.tolist(),
            "train_bugs": split["train"], "calibration_bugs": split["calibration"], "test_bugs": split["test"],
            "risk_threshold": threshold, "threshold_selection": {"max_cheap_harm": max_cheap_harm,
                "min_admitted": min_admitted, "admitted_rows": admitted_count,
                "observed_calibration_harm": observed_harm, "statistical_guarantee": False},
            "calibration_metrics_in_sample": metrics(cy, cp),
            "held_out_test_metrics": metrics(vy, vp), "held_out_uncalibrated_metrics": metrics(vy, raw_v),
            "target": target, "label_definition": rows[0]["label_definition"],
            "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "limitations": ["Candidate harm calibration does not certify issue-level early stopping.",
                            "Tiny-data calibration and threshold selection may overfit; use held-out bugs."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-cheap-harm", type=float, default=.05)
    p.add_argument("--min-admitted", type=int, default=10)
    p.add_argument("--target", choices=["candidate", "ranking"], default="candidate")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    artifact = train(rows, json.loads(args.split.read_text()), args.max_cheap_harm, args.min_admitted, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in artifact.items() if k not in {"booster", "features"}}, indent=2))


if __name__ == "__main__":
    main()
