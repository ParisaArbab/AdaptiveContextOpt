"""Offline comparison of compiler-parsed Java call syntax on real bug archives.

Use only pre-existing source snippets and failure evidence. A JDK AST verifies
invocation syntax and number of parameters, but target edges remain heuristic.
No gold patches or LLMs are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchor_budget import anchor_first_selection
from .java_ast_bridge import extract_ast_calls, verified_syntax_candidates
from .real_bug_closure_stress import ROOT, BUGS, BUDGETS, simple_name


def evaluate(root: Path = ROOT) -> dict:
    cases = []
    for bug in BUGS:
        source = root / bug / "input_snapshot.json"
        snapshot = json.loads(source.read_text(encoding="utf-8"))
        candidates = snapshot["candidates"]
        audit = extract_ast_calls(candidates)
        graph = verified_syntax_candidates(candidates, audit)
        edges = set(map(tuple, graph["edges"]))
        anchors = [
            i for i, item in enumerate(candidates)
            if simple_name(item["function_id"]) + "(" in snapshot["context"]
        ]
        costs = [int(c["features"]["l1_tokens"]) for c in candidates]
        rows = []
        for budget in BUDGETS:
            selected = anchor_first_selection(costs, anchors, edges, budget)
            rows.append({
                "budget_proxy": budget,
                "selected_methods": len(selected.selected),
                "selected_proxy_tokens": selected.tokens,
                "failure_named_methods_kept": len(selected.mandatory_anchors_kept),
                "failure_named_methods_total": len(anchors),
                "missing_candidate_call_edges": len(selected.unsatisfied_edges),
                "selected_ids": [candidates[i]["function_id"] for i in selected.selected],
            })
        cases.append({
            "bug": bug,
            "source_file": str(source),
            "source_candidates": len(candidates),
            "syntactic_calls": graph["calls_extracted"],
            "candidate_edges": graph["edges"],
            "ambiguous_calls": graph["ambiguous"],
            "unresolved_calls_count": len(graph["unresolved"]),
            "parse_error_candidates": graph["parse_error_indices"],
            "rows": rows,
        })
    return {
        "scope": "Two previously examined Defects4J development bugs; CPU AST syntax audit",
        "parser": "JDK JavacTask.parse on separately wrapped archived method snippets",
        "qualification": "Syntactic calls are parsed, but edges are only unique same-owner name+arity candidates, not type-resolved Java bindings",
        "gold_labels_used": False,
        "llm_executed": False,
        "budget_unit": "precomputed L1 sketch token proxy",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/java_ast_real_bug.json"))
    args = parser.parse_args()
    result = evaluate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(f"{case['bug']}: {case['syntactic_calls']} syntactic calls, "
              f"{len(case['candidate_edges'])} candidate edges, "
              f"{len(case['parse_error_candidates'])} parse-error snippets")
        for row in case["rows"]:
            print(f"  budget={row['budget_proxy']} "
                  f"failure-named={row['failure_named_methods_kept']}/"
                  f"{row['failure_named_methods_total']} "
                  f"missing={row['missing_candidate_call_edges']}")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
