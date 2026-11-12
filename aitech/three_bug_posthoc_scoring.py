"""Score frozen three-bug risk-gated selections with official patch methods.

Patch data is loaded only after generating selection results. This is
posthoc development scoring, not a held-out evaluation or LLM prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .risk_gated_pilot import experiment

LABELS = Path("data/aitech/three_bug_posthoc_patch_labels.json")


def score(selections: dict, labels: dict) -> dict:
    if not labels.get("scoring_only") or not labels.get("selection_frozen_before_labels"):
        raise ValueError("labels must be isolated to scoring")
    rows = []
    for case in selections["cases"]:
        bug = case["bug"]
        label_info = labels["bugs"][bug]
        gold = set(label_info["changed_methods"])
        if not gold:
            raise ValueError(f"missing gold labels: {bug}")
        for row in case["rows"]:
            selected = set(row["selected_ids"])
            hits = selected & gold
            rows.append({
                "bug": bug,
                "budget_proxy": row["budget_proxy"],
                "method": row["method"],
                "changed_method_hits": len(hits),
                "changed_method_total": len(gold),
                "changed_method_recall": len(hits) / len(gold),
                "retained_changed_methods": sorted(hits),
                "selected_proxy_tokens": row["used_proxy_tokens"],
            })
    by_group: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        by_group.setdefault((row["bug"], row["budget_proxy"]), {})[row["method"]] = row
    paired = []
    for (bug, budget), policies in sorted(by_group.items()):
        a, b = policies["anchor_first"], policies["risk_gated"]
        paired.append({
            "bug": bug, "budget_proxy": budget,
            "risk_minus_anchor_changed_method_recall": (
                b["changed_method_recall"] - a["changed_method_recall"]
            ),
            "anchor_first_recall": a["changed_method_recall"],
            "risk_gated_recall": b["changed_method_recall"],
        })
    return {
        "scope": "Three previously studied archived Defects4J bugs; post-selection changed-method recall",
        "patch_sources": {bug: labels["bugs"][bug]["patch"] for bug in labels["bugs"]},
        "selection_frozen_before_gold_scoring": True,
        "llm_ran": False,
        "rows": rows, "paired": paired,
        "num_paired_conditions": len(paired),
        "risk_better_than_anchor": sum(p["risk_minus_anchor_changed_method_recall"] > 0 for p in paired),
        "risk_worse_than_anchor": sum(p["risk_minus_anchor_changed_method_recall"] < 0 for p in paired),
        "limitations": labels.get("limitations", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/aitech/three_bug_posthoc.json"))
    args = parser.parse_args()
    # Freeze experimental selections before opening gold labels.
    selections = experiment()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    result = score(selections, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["paired"]:
        print(f"{row['bug']} {row['budget_proxy']}: "
              f"anchor={row['anchor_first_recall']:.2f}, "
              f"risk={row['risk_gated_recall']:.2f}, "
              f"delta={row['risk_minus_anchor_changed_method_recall']:+.2f}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
