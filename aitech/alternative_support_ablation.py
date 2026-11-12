"""Controlled ablation of alternative support bundles.

This is a synthetic theorem-checking/selector microcase, NOT an
empirical claim about actual Defects4J, learned dependency detection,
or LLM accuracy. Its purpose is to expose when strict union-of-support
overconstrains context relative to one-sufficient-path semantics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alternative_support import exact_select, dangling_evidence
from .evidence import Evidence, select_evidence

BUDGETS = (8, 10, 12, 14, 16)


def run() -> dict:
    items = [
        Evidence("failed_assertion", "expected true; got false", 4, 10),
        Evidence("trace_bridge", "test calls parseNumber", 4, 1),
        Evidence("function_definition", "parseNumber source", 3, 0.6),
        Evidence("assertion_context", "test constructs input", 3, 0.6),
        Evidence("unrelated_warning", "unrelated warning", 4, 5),
    ]
    # Two alternative support paths for interpreting failed_assertion.
    # Either a call bridge OR the pair definition+assertion context.
    alternatives = {
        "failed_assertion": (
            ("trace_bridge",),
            ("function_definition", "assertion_context"),
        )
    }
    strict_union = {
        "failed_assertion": (
            "trace_bridge", "function_definition", "assertion_context"
        )
    }
    results = []
    for budget in BUDGETS:
        for method in ("relevance_only", "all_edges_required",
                       "one_sufficient_path_exact"):
            if method == "one_sufficient_path_exact":
                out = exact_select(items, alternatives, budget)
                selected = list(out.ids)
                spent = out.tokens
                relevance = out.relevance
                combinations = out.evaluated_subsets
            else:
                out = select_evidence(
                    items, strict_union, budget,
                    preserve_dependencies=method == "all_edges_required")
                selected = list(out.ids)
                spent = out.tokens
                relevance = out.total_relevance
                combinations = None
            dangling = dangling_evidence(selected, alternatives)
            results.append({
                "budget": budget,
                "method": method,
                "selected_ids": selected,
                "spent_proxy_tokens": spent,
                "retained_relevance": relevance,
                "sufficient_support_missing": list(dangling),
                "failure_retained": "failed_assertion" in selected,
                "subsets_evaluated": combinations,
            })
    return {
        "scope": "Synthetic controlled microcase; not actual bug localization",
        "hypothesis": "OR-of-AND sufficient support can avoid costly forced union of all alternative paths",
        "source_of_support_sets": "Manually defined illustrative dependency alternatives",
        "support_sets": {k: [list(x) for x in v] for k, v in alternatives.items()},
        "budget_unit": "illustrative integer proxy tokens",
        "budgets": list(BUDGETS),
        "methods": ["relevance_only", "all_edges_required", "one_sufficient_path_exact"],
        "llm_executed": False,
        "fault_localization_accuracy_measured": False,
        "results": results,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=Path("results/aitech/alternative_support_microcase.json"))
    args = p.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["results"]:
        print(row["budget"], row["method"], "tokens=", row["spent_proxy_tokens"],
              "failure=", row["failure_retained"],
              "missing_support=", len(row["sufficient_support_missing"]))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
