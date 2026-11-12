"""Evaluate no-LLM, pre-gold source ranking as a mandatory CPU baseline.

This does not replace model learning; it is a diagnostic for models
that almost always select the first candidate. Model-free decisions
are frozen before posthoc Defects4J changed-class labels are opened.
It reports **incremental prompt and generation tokens = 0** but does
not pretend source parsing/retrieval computation is free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .failure_aligned_reranker import shortlist
from .test_name_anchored_reranker import hinted_shortlist
from .cpu_live_defects4j_pairs import _predicted_class


def pre_gold_predictions(snapshots: dict[str, dict]) -> list[dict]:
    frozen = []
    for bug, snapshot in sorted(snapshots.items()):
        if snapshot.get("provenance", {}).get("gold_patches_loaded") is not False:
            raise ValueError("gold-tainted source snapshot")
        candidates = snapshot["candidates"]
        alternatives = {
            "original_top8": list(range(min(8, len(candidates)))),
            "trace_idf": shortlist(snapshot, limit=8),
            "test_name_anchor": hinted_shortlist(snapshot, limit=8),
        }
        for name, order in alternatives.items():
            if not order:
                raise ValueError("empty candidate selection")
            frozen.append({
                "bug": bug, "method": candidates[order[0]]["function_id"],
                "condition": name, "top_indices": order,
                "failure_sha256": snapshot["provenance"]["failure_sha256"],
                "model_input_tokens": 0,
                "model_output_tokens": 0,
                "model_inference_executed": False,
                "retrieval_cpu_cost_not_measured": True,
            })
    return frozen


def score_frozen(frozen: list[dict], posthoc: dict) -> dict:
    labels = {x["bug"]: set(x["gold_modified_classes"])
              for x in posthoc.get("results", []) if not x.get("skipped")}
    if not labels:
        raise ValueError("missing posthoc labels")
    rows = []
    for row in frozen:
        if row["bug"] not in labels:
            raise ValueError("missing gold for frozen bug")
        rows.append({
            **row,
            "modified_class_hit_at_1": _predicted_class(row["method"])
                                       in labels[row["bug"]],
        })
    by_policy = {}
    for policy in ("original_top8", "trace_idf", "test_name_anchor"):
        part = [r for r in rows if r["condition"] == policy]
        by_policy[policy] = {
            "n": len(part),
            "changed_class_hits_at_1": sum(r["modified_class_hit_at_1"] for r in part),
            "total_model_input_tokens": 0,
            "total_model_output_tokens": 0,
        }
    return {
        "scope": "Previously examined Math-1/Chart-1 development-only",
        "model_inference_executed": False,
        "gold_read_posthoc": True,
        "not_root_cause_proof": True,
        "model_free_baselines": by_policy,
        "rows": rows,
        "retrieval_cpu_time_measured": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", action="append", nargs=2,
                   metavar=("BUG", "PATH"), required=True)
    p.add_argument("--posthoc", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    snaps = {bug: json.loads(Path(path).read_text())
             for bug, path in args.snapshot}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = pre_gold_predictions(snaps)
    (args.output_dir / "pre_gold_model_free_predictions.json").write_text(
        json.dumps(frozen, indent=2) + "\n")
    posthoc = json.loads(args.posthoc.read_text())
    result = score_frozen(frozen, posthoc)
    (args.output_dir / "posthoc_model_free_scores.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["model_free_baselines"], indent=2))


if __name__ == "__main__":
    main()
