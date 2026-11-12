"""Strong-baseline audit for new multi-resolution evidence method.

Score *frozen* outputs of the new proposed policy against existing
tiered strategies that were implemented separately. Gold patch labels
are read only for posthoc diagnostics, never for selecting evidence.

This analysis can reject a candidate method. It is not model accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tiered_evidence_pilot import experiment as existing_experiment


def compare(frozen: dict, existing: dict, labels: dict) -> dict:
    if frozen.get("gold_used") is not False or labels.get("scoring_only") is not True:
        raise ValueError("frozen selections and scoring-only labels required")
    original = {}
    for case in existing["cases"]:
        for row in case["rows"]:
            original[(case["bug"], row["budget_proxy"], row["method"])] = row
    rows = []
    for case in frozen["cases"]:
        bug = case["bug"]
        gold = set(labels["bugs"][bug]["changed_methods"])
        for row in case["rows"]:
            budget = row["budget_proxy"]
            new_l1 = set(row["tiered_l1_ids"])
            new_any = new_l1 | set(row["tiered_l0_ids"])
            score_new = (len(new_any & gold), len(new_l1 & gold))
            controls = {}
            for policy in ("l1_only", "tiered_upgrade", "anchor_detail_first", "anchor_rank_detail"):
                control = original[(bug, budget, policy)]
                tiers = control["candidate_id_tier"]
                identity = len(gold & set(tiers))
                detailed = sum(tiers.get(g) == "L1" for g in gold)
                controls[policy] = {
                    "identity_hits": identity,
                    "detail_hits": detailed,
                    "tokens": control["used_proxy_tokens"],
                }
            best_detail = max(x["detail_hits"] for x in controls.values())
            rows.append({
                "bug": bug,
                "budget_proxy": budget,
                "gold_total": len(gold),
                "new_identity_hits": score_new[0],
                "new_l1_detail_hits": score_new[1],
                "new_proxy_tokens": row["tiered_tokens"],
                "existing_controls": controls,
                "best_existing_l1_hits": best_detail,
                "new_better_than_best_existing_l1": score_new[1] > best_detail,
                "new_worse_than_best_existing_l1": score_new[1] < best_detail,
            })
    return {
        "scope": "Development-only strong-baseline assessment on previously used bugs",
        "gold_labels_only_posthoc": True,
        "llm_executed": False,
        "cases": rows,
        "new_detail_wins_over_best_existing": sum(r["new_better_than_best_existing_l1"] for r in rows),
        "new_detail_losses_to_best_existing": sum(r["new_worse_than_best_existing_l1"] for r in rows),
        "new_detail_ties": sum(r["new_l1_detail_hits"] == r["best_existing_l1_hits"] for r in rows),
        "limitations": [
            "These bugs and gold labels were previously examined, so not clean held-out",
            "New and existing methods may use different L0 packing choices",
            "L1 detail is not bug localization or patch correctness",
            "Evidence token costs are stored proxies, not model tokenizer counts",
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
    existing = existing_experiment()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    results = compare(frozen, existing, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for row in results["cases"]:
        print(row["bug"], row["budget_proxy"],
              "new detail", row["new_l1_detail_hits"],
              "strongest previous", row["best_existing_l1_hits"])
    print("Saved", args.output)


if __name__ == "__main__":
    main()
