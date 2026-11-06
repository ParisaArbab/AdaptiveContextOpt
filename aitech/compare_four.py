"""Combine a historical three-arm run with the new provenance-only condition.

The earlier run is reused only if model, seed, issue, instance, and budgets
match. This avoids wasting expensive Ollama calls.

Usage:
 python -m aitech.compare_four \
  --previous results/aitech/sympy20590_ollama_three_arm.json \
  --provenance results/aitech/sympy20590_provenance_only.json \
  --output results/aitech/sympy20590_four_arm_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = ("raw_excerpt", "relevance_only", "dependency_aware")
EXTRA = "dependency_aware_provenance"
FIELDS = ("instance", "model", "seed", "temperature", "budget_proxy",
          "raw_proxy_tokens")


def compare(previous: dict, provenance: dict) -> dict:
    for field in FIELDS:
        if previous.get(field) != provenance.get(field):
            raise ValueError(f"incompatible experiments: {field}")
    previous_arms = previous.get("conditions", {})
    new_arms = provenance.get("conditions", {})
    if any(arm not in previous_arms for arm in BASE):
        raise ValueError("previous experiment must contain original three arms")
    if EXTRA not in new_arms:
        raise ValueError("provenance experiment missing fourth arm")
    if (previous_arms["dependency_aware"]["selected_ids"]
            != new_arms[EXTRA]["selected_ids"]):
        raise ValueError("selected evidence differs between dependency-aware arms")
    merged = {name: previous_arms[name] for name in BASE}
    merged[EXTRA] = new_arms[EXTRA]
    rows = []
    for name, data in merged.items():
        score = data.get("offline_score", {})
        rows.append({
            "method": name,
            "file_hit_1": score.get("file_hit_1"),
            "entity_hit_1": score.get("entity_hit_1"),
            "file": data["prediction"]["file"],
            "function": data["prediction"]["function"],
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "selected_tokens_proxy": data.get("tokens_proxy"),
            "dangling_count_known": len(data.get("dangling_ids", [])),
            "prompt_chars": data.get("prompt_chars"),
        })
    return {
        "experiment": "archived SymPy real-source seven-line microcase",
        "instance": previous["instance"],
        "model": previous["model"],
        "seed": previous["seed"],
        "temperature": previous["temperature"],
        "source_runs": "previous three-arm run + new provenance-only run",
        "warning": "Provenance metadata adds prompt tokens; C/D are NOT total-token matched.",
        "gold_used_at_inference": False,
        "conditions": merged,
        "summary": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.previous.read_text(encoding="utf-8")),
        json.loads(args.provenance.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for item in result["summary"]:
        print(f"{item['method']:29s}  file={str(item['file_hit_1']):5s} "
              f"entity={str(item['entity_hit_1']):5s} "
              f"in={str(item['input_tokens']):>5s} "
              f"out={str(item['output_tokens']):>4s} "
              f"seconds={str(item['elapsed_seconds']):>8s} "
              f"pred={item['file']}::{item['function']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
