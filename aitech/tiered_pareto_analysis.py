"""Pareto diagnostic: more method IDs must not conceal lost source detail.

Evaluate tiered policies against fixed L1-only evidence in the same
bug and budget. Identity recall and L1 changed-method recall are separate
objectives. This describes observed development data, not causal proof,
model accuracy or a statistical generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .score_tiered_posthoc import score
from .tiered_evidence_pilot import experiment
from .three_bug_posthoc_scoring import LABELS

METHODS = ("tiered_l0", "tiered_upgrade", "anchor_detail_first", "anchor_rank_detail")


def analyze(scoring: dict) -> dict:
    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for row in scoring["rows"]:
        key=(row["bug"],row["budget_proxy"])
        methods=by_key.setdefault(key,{})
        if row["method"] in methods:
            raise ValueError("duplicate condition")
        methods[row["method"]]=row
    rows=[]
    for (bug,budget), methods in sorted(by_key.items()):
        if set(methods) != {"l1_only",*METHODS}:
            raise ValueError(f"missing policy for {bug} budget {budget}")
        base=methods["l1_only"]
        for method in METHODS:
            alt=methods[method]
            id_delta=alt["identity_recall"]-base["identity_recall"]
            detail_delta=alt["detailed_source_recall"]-base["detailed_source_recall"]
            rows.append({
                "bug":bug,
                "budget_proxy":budget,
                "method":method,
                "identity_recall_gain":id_delta,
                "l1_detail_recall_gain":detail_delta,
                "weakly_pareto_dominates_l1_only":id_delta>=0 and detail_delta>=0,
                "strictly_pareto_improves_l1_only":id_delta>=0 and detail_delta>=0 and
                    (id_delta>0 or detail_delta>0),
                "identity_gain_with_detail_loss":id_delta>0 and detail_delta<0,
                "model_localization_measured":False,
            })
    counts={}
    for method in METHODS:
        subset=[r for r in rows if r["method"]==method]
        counts[method]={
            "conditions":len(subset),
            "weak_dominance":sum(r["weakly_pareto_dominates_l1_only"] for r in subset),
            "strict_improvement":sum(r["strictly_pareto_improves_l1_only"] for r in subset),
            "identity_gain_detail_loss":sum(r["identity_gain_with_detail_loss"] for r in subset),
        }
    return {
        "scope":"Development-only three-bug archived matched-proxy tier evaluation",
        "source_data_is_frozen":True,
        "llm_ran":False,
        "metrics":"posthoc official patch changed method L0 identifier presence and L1 source presence",
        "not_measured":["causal localization","LLM prediction quality","actual model tokenizer cost"],
        "method_counts":counts,
        "rows":rows,
    }


def render(summary: dict) -> str:
    lines=[
        "# AITECH: hierarchical code evidence Pareto comparison",
        "",
        "Three previously studied Defects4J bugs, four frozen L0/L1 proxy budgets.",
        "Strictly separate name/signature coverage from detailed L1 code coverage.",
        "A weak Pareto improvement means no lower changed-method coverage on either",
        "measure at the same budget; it is NOT a measured debugging accuracy gain.",
        "",
        "| Policy | Conditions | No-worse on both | Strict coverage improvement | More IDs but less detailed code |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy,count in summary["method_counts"].items():
        lines.append(
            f"| {policy} | {count['conditions']} | {count['weak_dominance']} "
            f"| {count['strict_improvement']} | {count['identity_gain_detail_loss']} |"
        )
    lines.extend(["","## Interpretation and limitations","",
        "All bugs have been used for method development; no unseen test set.",
        "L0 evidence provides names and signatures, not executable or causal source.",
        "L1 is a lossy code sketch; its token costs are stored proxies.",
        "Official patch changed methods are imperfect localization labels.",
        "No LLM was run as part of this Pareto study, so no AI debugging",
        "accuracy gain or novelty over existing research has been established.",
    ])
    return "\n".join(lines)+"\n"


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("results/aitech/tiered_pareto"))
    args=parser.parse_args()
    selections=experiment()
    labels=json.loads(LABELS.read_text(encoding="utf-8"))
    scoring=score(selections,labels)
    summary=analyze(scoring)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"results.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (args.output_dir/"report.md").write_text(render(summary),encoding="utf-8")
    print(render(summary))


if __name__=="__main__":
    main()
