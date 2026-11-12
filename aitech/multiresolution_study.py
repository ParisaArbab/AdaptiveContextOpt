"""Controlled multi-resolution vs single-resolution evidence ablation.

Reads only archived buggy-source candidate snapshots, no gold patches.
Single-resolution baseline reports L1/full-structure anchors. Multi-
resolution reports separate L0 identity and L1 structural anchor retention,
so signature-only evidence is never misreported as full code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .failure_conditioned_selector import select
from .failure_conditioned_study import BUG_INPUTS, BUDGETS, SPLITS
from .java_call_audit import sketch_call_audit
from .multiresolution_selector import select_tiered


def experiment() -> dict:
    cases = []
    for bug, path in BUG_INPUTS.items():
        raw = path.read_bytes()
        snapshot = json.loads(raw)
        candidates = snapshot["candidates"]
        audit = sketch_call_audit(candidates)
        graph = set(audit.edges)
        rows = []
        for budget in BUDGETS:
            single = select(candidates, snapshot["context"], budget,
                            method="failure_graph", edges=graph)
            multi = select_tiered(candidates, snapshot["context"], budget,
                                  edges=graph, graph_confidence="lexical")
            rows.append({
                "budget_proxy": budget,
                "single_l1_tokens": single.proxy_tokens,
                "single_selected_l1": len(single.selected),
                "single_anchor_l1": single.anchors_kept,
                "single_anchor_total": single.anchors_total,
                "single_relevance_proxy": single.evidence_score,
                "tiered_tokens": multi.proxy_tokens,
                "tiered_l0_identity_methods": multi.l0_count,
                "tiered_l1_structure_methods": multi.l1_count,
                "tiered_anchor_identities": multi.anchor_identities_retained,
                "tiered_anchor_l1": multi.anchor_l1_retained,
                "tiered_anchor_total": multi.anchors_total,
                "tiered_relevance_fidelity_proxy": multi.weighted_proxy_utility,
                "tiered_levels": list(multi.levels),
                "single_selected_ids": list(single.ids),
                "tiered_l0_ids": [candidates[i]["function_id"]
                                 for i, level in enumerate(multi.levels) if level == 1],
                "tiered_l1_ids": [candidates[i]["function_id"]
                                 for i, level in enumerate(multi.levels) if level == 2],
            })
        cases.append({
            "bug": bug,
            "split": SPLITS[bug],
            "source_path": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "lexical_candidate_edges": len(graph),
            "rows": rows,
        })
    return {
        "research_question": "Can adaptive L0/L1 fidelity preserve failure context without the high opportunity cost of mandatory full-code anchors?",
        "scope": "Three archived Defects4J bug snapshots, CPU-only evidence allocation",
        "gold_used": False,
        "llm_ran": False,
        "budget_unit": "precomputed sketch token estimates",
        "utility": "fixed relevance × assumed L0 0.20 / L1 1.00 fidelity, not accuracy",
        "evaluation_not_clean_unseen": True,
        "limitations": [
            "L0 summaries can only identify methods, not capture their code logic",
            "Multi-resolution utility weights are hypotheses, not measured model value",
            "Token proxies are not true model tokenizer counts",
            "Lexical edges may be incorrect and graph benefit remains unproven",
            "No held-out independent repository, labels, or actual LLM localization",
        ],
        "cases": cases,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(
        "results/aitech/multiresolution_three_bug.json"))
    args = p.parse_args()
    result = experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(case["bug"], case["split"])
        for row in case["rows"]:
            print(row["budget_proxy"],
                  "single L1", row["single_selected_l1"],
                  "single anchor L1", row["single_anchor_l1"],
                  "tiered L0", row["tiered_l0_identity_methods"],
                  "tiered L1", row["tiered_l1_structure_methods"],
                  "tiered anchor ID", row["tiered_anchor_identities"],
                  "tiered anchor L1", row["tiered_anchor_l1"])
    print("Saved", args.output)


if __name__ == "__main__":
    main()
