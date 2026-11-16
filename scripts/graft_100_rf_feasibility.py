"""Leakage-safe feasibility check for an adaptive Random Forest graph gate.

This is a *separate research analysis*, not a replacement for the frozen run.
Never use gold-file ranks or graph outcomes as prediction features.
When there are no positive graph-gain labels, do NOT fit a fake classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.graft_100_postmortem import diagnose


def collect_development_labels(audit: dict, fusion: str = "rrf"):
    rows = [r for r in audit["paired"]
            if r["split"] == "development" and r["fusion"] == fusion]
    # Deliberately no held-out evaluation rows in training.
    labels = [{"bug": r["bug"], "label": int(r["mrr_delta_focused"] > 0),
               "gain": r["mrr_delta_focused"]} for r in rows]
    return labels


def feasibility(root: Path, fusion: str = "rrf") -> dict:
    audit = diagnose(root)
    labels = collect_development_labels(audit, fusion)
    counts = Counter(r["label"] for r in labels)
    result = {
        "schema": "graft-100-rf-feasibility-v1",
        "selection_sha256": audit["selection_sha256"],
        "fusion": fusion,
        "training_split": "development_only",
        "evaluation_split_used_for_training": False,
        "development_pairs": len(labels),
        "label_definition": "graph-focused file-MRR > L1 file-MRR for same bug and fusion",
        "label_counts": {str(k): v for k, v in sorted(counts.items())},
        "learnable": len(counts) >= 2 and min(counts.values()) >= 3,
        "trained": False,
        "reason": "",
        "development_labels": labels,
    }
    if len(counts) < 2:
        result["reason"] = (
            "All observed development graph gains have the same binary label. "
            "Cannot train or validate an honest Random Forest gain predictor."
        )
    elif min(counts.values()) < 3:
        result["reason"] = (
            "Too few examples in the minority class for a useful independent "
            "development train/calibration split."
        )
    else:
        result["reason"] = (
            "Label diversity exists, but pre-graph prediction features must be "
            "extracted and independently validated before any model is fitted."
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/graft_100_cohort_20261115"))
    parser.add_argument("--out", type=Path, default=Path("results/graft_100_recovery_audit"))
    parser.add_argument("--fusion", choices=("rrf", "suspicion_rank_blend"), default="rrf")
    args = parser.parse_args()
    outcome = feasibility(args.root, args.fusion)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ("rf_feasibility_" + args.fusion + ".json")
    path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(outcome["reason"])
    print("Trained:", outcome["trained"], "Development labels:", outcome["label_counts"])
    print("Wrote", path)


if __name__ == "__main__":
    main()
