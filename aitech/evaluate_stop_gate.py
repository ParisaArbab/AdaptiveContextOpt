"""Evaluate exact test-name STOP versus ESCALATE on frozen buggy snapshots.

Two phase:
  (1) freeze class-hint predictions from pre-gold source snapshots;
  (2) open an existing posthoc modified-class record for scoring.
No Qwen run is performed here. An ESCALATE decision is deliberately
not scored as a hit or simulated LLM success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_free_stop_gate import decide_model_free_stop


def freeze(snapshots: dict[str, dict]) -> list[dict]:
    records = []
    for bug, snapshot in sorted(snapshots.items()):
        decision = decide_model_free_stop(snapshot)
        records.append({
            "bug": bug, "action": decision.action,
            "predicted_class": decision.predicted_class,
            "candidate_method": decision.candidate_method,
            "reason": decision.reason,
            "model_input_tokens": decision.model_input_tokens,
            "model_output_tokens": decision.model_output_tokens,
            "failed_test_source_sha256": snapshot["provenance"]["failure_sha256"],
        })
    return records


def score(records: list[dict], posthoc: dict) -> dict:
    gold = {
        row["bug"]: set(row["gold_modified_classes"])
        for row in posthoc["results"] if not row.get("skipped")
    }
    results = []
    for row in records:
        if row["bug"] not in gold:
            raise ValueError("missing gold for bug")
        scored = {**row}
        if row["action"] == "STOP_CLASS_HINT":
            scored["class_hit_at_1"] = row["predicted_class"] in gold[row["bug"]]
        else:
            scored["class_hit_at_1"] = None
        results.append(scored)
    stopped = [r for r in results if r["action"] == "STOP_CLASS_HINT"]
    return {
        "scope": "Development-only Math-1/Chart-1, not heldout generalization",
        "model_executed": False,
        "stopped_bugs": len(stopped),
        "total_bugs": len(results),
        "stop_class_hits_at_1": sum(r["class_hit_at_1"] for r in stopped),
        "llm_input_tokens_for_stop": 0,
        "retrieval_and_analysis_cpu_time_unmeasured": True,
        "heuristic_not_calibrated": True,
        "results": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", action="append", nargs=2,
                   metavar=("BUG", "PATH"), required=True)
    p.add_argument("--posthoc", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    snapshots = {bug: json.loads(Path(path).read_text())
                 for bug, path in args.snapshot}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = freeze(snapshots)
    (args.output_dir / "pre_gold_decisions.json").write_text(
        json.dumps(frozen, indent=2) + "\n")
    scores = score(frozen, json.loads(args.posthoc.read_text()))
    (args.output_dir / "posthoc_scores.json").write_text(
        json.dumps(scores, indent=2) + "\n")
    print("STOP", scores["stopped_bugs"], "of", scores["total_bugs"],
          "bugs; correct changed class", scores["stop_class_hits_at_1"])


if __name__ == "__main__":
    main()
