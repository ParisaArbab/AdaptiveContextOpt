"""Posthoc changed-method coverage for FROZEN multi-resolution selections.

Run the selector BEFORE reading labels. Score identity (L0 or L1) and
detailed source (L1 only) separately, against official Defects4J patch
method labels manually mapped in the existing repository. This is
development-only diagnostic evidence, not LLM localization accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(frozen: dict, labels: dict) -> dict:
    if frozen.get("gold_used") is not False:
        raise ValueError("selections must explicitly exclude gold patches")
    if labels.get("scoring_only") is not True:
        raise ValueError("labels must be scoring-only")
    rows = []
    for case in frozen["cases"]:
        bug = case["bug"]
        gold = set(labels["bugs"][bug]["changed_methods"])
        if not gold:
            raise ValueError("missing patch labels")
        for row in case["rows"]:
            single = set(row["single_selected_ids"])
            low = set(row["tiered_l0_ids"])
            high = set(row["tiered_l1_ids"])
            if low & high:
                raise ValueError("L0 and L1 must be disjoint")
            selected = low | high
            baseline_hits = single & gold
            detailed_hits = high & gold
            identity_hits = selected & gold
            rows.append({
                "bug": bug,
                "split": case["split"],
                "budget_proxy": row["budget_proxy"],
                "changed_methods_total": len(gold),
                "baseline_l1_hits": len(baseline_hits),
                "tiered_identity_hits": len(identity_hits),
                "tiered_l1_detail_hits": len(detailed_hits),
                "identity_gain_vs_single": len(identity_hits) - len(baseline_hits),
                "detail_gain_vs_single": len(detailed_hits) - len(baseline_hits),
                "baseline_tokens": row["single_l1_tokens"],
                "tiered_tokens": row["tiered_tokens"],
                "selected_gold_L0_only": sorted((low & gold)),
                "selected_gold_L1": sorted(detailed_hits),
                "missed_gold": sorted(gold - selected),
            })
    return {
        "scope": "Exploratory posthoc patch-method coverage on three previously studied bugs",
        "selection_source": "frozen multiresolution_study output",
        "gold_used_only_for_scoring": True,
        "llm_ran": False,
        "rows": rows,
        "identity_gain_conditions": sum(x["identity_gain_vs_single"] > 0 for x in rows),
        "detail_gain_conditions": sum(x["detail_gain_vs_single"] > 0 for x in rows),
        "detail_loss_conditions": sum(x["detail_gain_vs_single"] < 0 for x in rows),
        "limitations": [
            "Changed methods can differ from true fault cause",
            "All three bugs previously examined, not independent final test",
            "L0 identity coverage is not detailed source coverage",
            "Budget based on L0/L1 proxy counts not actual prompt tokens",
            "No LLM inference or patch validation",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--labels", type=Path,
                   default=Path("data/aitech/three_bug_posthoc_patch_labels.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    result = score(frozen, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        print(row["bug"], row["budget_proxy"],
              "baseline detailed", row["baseline_l1_hits"],
              "tiered identity", row["tiered_identity_hits"],
              "tiered detailed", row["tiered_l1_detail_hits"])
    print("Saved", args.output)


if __name__ == "__main__":
    main()
