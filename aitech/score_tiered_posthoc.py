"""Evaluate frozen tiered selectors against official changed-method labels.

Selection is generated before opening patch labels. Report L0 method
identity and L1 detailed-code coverage separately, not as LLM accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tiered_evidence_pilot import experiment
from .three_bug_posthoc_scoring import LABELS


def score(selections: dict, labels: dict) -> dict:
    if labels.get("scoring_only") is not True or labels.get("selection_frozen_before_labels") is not True:
        raise ValueError("labels must be reserved for post-selection scoring")
    rows=[]
    for case in selections["cases"]:
        bug=case["bug"]
        gold=set(labels["bugs"][bug]["changed_methods"])
        if not gold:
            raise ValueError("missing patch methods for "+bug)
        for selected in case["rows"]:
            tiers=selected["candidate_id_tier"]
            any_hit=gold.intersection(tiers)
            l1_hit={identifier for identifier in gold if tiers.get(identifier)=="L1"}
            rows.append({
                "bug":bug,
                "budget_proxy":selected["budget_proxy"],
                "method":selected["method"],
                "changed_methods_total":len(gold),
                "changed_methods_at_any_tier":len(any_hit),
                "changed_methods_with_l1_detail":len(l1_hit),
                "identity_recall":len(any_hit)/len(gold),
                "detailed_source_recall":len(l1_hit)/len(gold),
                "used_proxy_tokens":selected["used_proxy_tokens"],
                "selected_tiers":dict(tiers),
            })
    groups={}
    for row in rows:
        groups.setdefault((row["bug"],row["budget_proxy"]),{})[row["method"]]=row
    paired=[]
    for (bug,budget), group in sorted(groups.items()):
        if set(group)!={"l1_only","tiered_l0","tiered_upgrade","anchor_detail_first"}:
            raise ValueError("missing tiered comparison policy")
        baseline=group["l1_only"]
        for policy in ("tiered_l0","tiered_upgrade","anchor_detail_first"):
            current=group[policy]
            paired.append({
                "bug":bug,"budget_proxy":budget,"method":policy,
                "identity_recall_gain":current["identity_recall"]-baseline["identity_recall"],
                "detailed_source_recall_gain":current["detailed_source_recall"]-baseline["detailed_source_recall"],
            })
    return {
        "scope":"Three previously studied Defects4J bugs, post-selection proxy-budget evaluation",
        "selection_frozen_before_scoring":True,
        "model_inference_performed":False,
        "identity_coverage_is_not_localization":True,
        "rows":rows,
        "paired":paired,
        "identity_gain_conditions":sum(x["identity_recall_gain"]>0 for x in paired),
        "detail_gain_conditions":sum(x["detailed_source_recall_gain"]>0 for x in paired),
        "detail_loss_conditions":sum(x["detailed_source_recall_gain"]<0 for x in paired),
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results/aitech/tiered_posthoc.json"))
    args=parser.parse_args()
    frozen=experiment()
    labels=json.loads(LABELS.read_text(encoding="utf-8"))
    results=score(frozen,labels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    for row in results["rows"]:
        print(f"{row['bug']:8} {row['budget_proxy']:4} {row['method']:14} "
              f"ID={row['changed_methods_at_any_tier']}/{row['changed_methods_total']} "
              f"L1={row['changed_methods_with_l1_detail']}/{row['changed_methods_total']} "
              f"cost={row['used_proxy_tokens']}")
    print("Saved",args.output)


if __name__=="__main__":
    main()
