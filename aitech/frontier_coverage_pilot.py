"""Reproducible diagnostic of cross-class code retrieval gaps.

Read the same three frozen archived Defects4J candidate pools as the
risk-gated pilot, without model inference, patch labels or network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .frontier_coverage import inspect_frontier
from .risk_gated_pilot import SOURCES


def experiment(sources: dict[str, Path] = SOURCES) -> dict:
    records = []
    for bug, path in sources.items():
        raw = path.read_bytes()
        candidates = json.loads(raw)["candidates"]
        audit = inspect_frontier(candidates)
        records.append({
            "bug": bug,
            "source_path": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "ranked_method_count": len(candidates),
            "qualified_calls_inspected": audit.qualified_calls,
            "qualified_candidate_edges": len(audit.in_pool),
            "missing_methods_in_ranked_owners": list(audit.missing_ranked_method),
            "outside_ranked_owner_references": list(audit.outside_ranked_owner),
            "ambiguous_qualified_references": list(audit.ambiguous),
            "candidate_edges": [
                [candidates[src]["function_id"], candidates[dst]["function_id"]]
                for src, dst in audit.in_pool
            ],
        })
    return {
        "scope": "three previously studied Defects4J candidate pools, exploratory source-retrieval diagnostic",
        "gold_patch_used": False,
        "llm_executed": False,
        "edge_quality": "lexical class-qualified CALL hints, not compiler-resolved graph",
        "cases": records,
        "next_action": (
            "Retrieve missing class-qualified method definitions using a "
            "complete pinned checkout, then verify symbol bindings and "
            "evaluate actual end-to-end localization at matched model-token budgets."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/frontier_coverage.json"))
    args = parser.parse_args()
    result = experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(f"{case['bug']}: qualified={case['qualified_calls_inspected']} "
              f"matched_edges={case['qualified_candidate_edges']} "
              f"missing_ranked_method={len(case['missing_methods_in_ranked_owners'])} "
              f"outside_owner={len(case['outside_ranked_owner_references'])} "
              f"ambiguous={len(case['ambiguous_qualified_references'])}")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
