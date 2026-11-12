"""CPU-only pilot: does risk-gated support beat unconditional graph expansion?

Two archived development Defects4J cases are used for code sanity and
the third case, Time-25, is a frozen exploratory transfer case. It is
NOT a true unseen benchmark holdout: Time-25 has been studied in earlier
work. Gold patches and models are never loaded.

Graph confidence is an UNCALIBRATED policy design choice: L1 lexical
calls 0.4, sketch arity matches 0.8, compiler verified would be 1.0.
Only the arity-sketch links are allowed to promote evidence by default.
Matched proxy budgets measure evidence packing, not model accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .anchor_budget import anchor_first_selection
from .java_call_audit import sketch_call_audit
from .risk_gated_context import Edge, select_context

SOURCES = {
    "Lang-1": Path("results/rave_fl_iteration_d/independent_live/Lang-1/input_snapshot.json"),
    "Time-1": Path("results/rave_fl_iteration_d/independent_live/Time-1/input_snapshot.json"),
    "Time-25": Path("results/rave_fl_rank_repair/live_original/Time-25/input_snapshot.json"),
}
BUDGETS = (200, 300, 450, 600)
METHODS = ("rank_fill", "anchor_first", "always_expand", "risk_gated")


def name(identifier: str) -> str:
    return identifier.split("(", 1)[0].rsplit(".", 1)[-1]


def run_case(snapshot: dict, bug: str) -> dict:
    candidates = snapshot["candidates"]
    ids = [c["function_id"] for c in candidates]
    costs = [int(c["features"]["l1_tokens"]) for c in candidates]
    anchors = [i for i, id_ in enumerate(ids)
               if name(id_) + "(" in snapshot["context"]]
    audit = sketch_call_audit(candidates)
    edges = tuple(Edge(src, dst, .8, "arity_sketch")
                  for src, dst in audit.edges)
    rows = []
    for budget in BUDGETS:
        for method in METHODS:
            if method == "rank_fill":
                chosen, used = set(), 0
                for i, cost in enumerate(costs):
                    if cost + used <= budget:
                        chosen.add(i)
                        used += cost
                graph_actions = 0
            elif method in ("anchor_first", "always_expand"):
                baseline = anchor_first_selection(
                    costs, anchors, audit.edges if method == "always_expand" else (),
                    budget)
                chosen = set(baseline.selected)
                used = baseline.tokens
                graph_actions = 0
            else:
                selection = select_context(
                    costs, anchors, edges, budget,
                    min_confidence=.75, graph_weight=1.)
                chosen = set(selection.selected)
                used = selection.tokens
                graph_actions = sum(
                    d["reason"] == "graph_support" for d in selection.decisions
                )
            rows.append({
                "bug": bug, "budget_proxy": budget,
                "method": method, "used_proxy_tokens": used,
                "selected_count": len(chosen),
                "failure_name_anchors_kept": len(chosen.intersection(anchors)),
                "failure_name_anchors_total": len(anchors),
                "missing_arity_sketch_edges": sum(
                    src in chosen and dst not in chosen for src, dst in audit.edges
                ),
                "graph_promoted_choices": graph_actions,
                "selected_ids": [ids[i] for i in sorted(chosen)],
            })
    return {
        "bug": bug,
        "candidate_count": len(candidates),
        "anchor_count": len(anchors),
        "arity_sketch_edges": len(audit.edges),
        "inspected_l1_call_lines": audit.inspected_calls,
        "rows": rows,
    }


def experiment(sources: dict[str, Path] = SOURCES) -> dict:
    cases = []
    hashes = {}
    for bug, path in sources.items():
        raw = path.read_bytes()
        hashes[bug] = {
            "source_path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        cases.append(run_case(json.loads(raw), bug))
    return {
        "research_question": "Does cost-aware, confidence-gated expansion improve anchor preservation vs always expanding graph support?",
        "scope": "Archived real Defects4J candidates, selector-only CPU comparison",
        "development_cases": ["Lang-1", "Time-1"],
        "transfer_case_previously_studied_not_heldout": ["Time-25"],
        "methods": list(METHODS),
        "budget_unit": "stored L1 sketch token proxies, not LLM tokens",
        "edge_provenance": "same-owner arity matching from lossy L1 sketches, not verified compiler calls",
        "edge_confidence": .8,
        "edge_confidence_calibrated": False,
        "policy_hyperparameters_frozen": {
            "min_confidence": .75, "graph_weight": 1.,
            "rank_weight": 1.,
        },
        "gold_labels_read": False,
        "llm_ran": False,
        "sources": hashes,
        "cases": cases,
        "limitations": [
            "Three bugs previously used in this project; no independent unseen split",
            "No gold file/function hit or LLM localization evaluated",
            "Graph confidences and rank utilities are heuristics, not learned/calibrated",
            "The smaller context may remove useful code despite keeping failure-name anchors",
            "Whole-sketch proxy costs do not equal actual prompt tokens",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/risk_gated_pilot.json"))
    args = parser.parse_args()
    result = experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(f"{case['bug']}: {case['candidate_count']} candidate methods, "
              f"{case['arity_sketch_edges']} audited arity edges")
        for row in case["rows"]:
            print(f"  {row['budget_proxy']:3} {row['method']:13} "
                  f"anchors {row['failure_name_anchors_kept']}/{row['failure_name_anchors_total']}, "
                  f"unresolved {row['missing_arity_sketch_edges']}, "
                  f"used {row['used_proxy_tokens']}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
