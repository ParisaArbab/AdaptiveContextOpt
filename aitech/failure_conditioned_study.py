"""Three-bug development/held-out evidence selection experiment.

Development: Lang-1, Time-1. Held-out evaluation: Time-25, whose archived
candidate inputs are a separate bug, but not an unseen project.

NO gold patches are read. NO LLM is called. Metrics measure only
failure-name anchors and dependency support, not localization accuracy.
All four arms receive the same candidate pool, budget, and source sketch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .failure_conditioned_selector import select
from .java_call_audit import sketch_call_audit

BUG_INPUTS = {
    "Lang-1": Path("results/rave_fl_iteration_d/independent_live/Lang-1/input_snapshot.json"),
    "Time-1": Path("results/rave_fl_iteration_d/independent_live/Time-1/input_snapshot.json"),
    "Time-25": Path("results/rave_fl_rank_repair/live_original/Time-25/input_snapshot.json"),
}
SPLITS = {"Lang-1": "development", "Time-1": "development", "Time-25": "evaluation"}
POLICIES = ("rank_fill", "anchor_first", "relevance_density", "failure_graph")
BUDGETS = (200, 300, 450, 600)


def experiment(inputs: dict[str, Path] = BUG_INPUTS) -> dict:
    cases = []
    for bug, path in inputs.items():
        raw = path.read_bytes()
        snapshot = json.loads(raw)
        candidates = snapshot["candidates"]
        graph = sketch_call_audit(candidates)
        edges = set(graph.edges)
        rows = []
        for budget in BUDGETS:
            for policy in POLICIES:
                result = select(
                    candidates, snapshot["context"], budget,
                    method=policy, edges=edges, graph_confidence="lexical")
                rows.append({
                    "budget_proxy": budget,
                    "method": policy,
                    "selected_ids": list(result.ids),
                    "selected_count": len(result.selected),
                    "used_proxy_tokens": result.proxy_tokens,
                    "failure_anchors_kept": result.anchors_kept,
                    "failure_anchors_total": result.anchors_total,
                    "unresolved_lexical_edges": result.unresolved_edges,
                    "retained_relevance_proxy": result.evidence_score,
                })
        cases.append({
            "bug": bug,
            "split": SPLITS.get(bug, "external"),
            "raw_source_path": str(path),
            "raw_source_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_count": len(candidates),
            "graph_provenance": "lossy L1 CALL entries, unique same-owner/name/arity heuristic",
            "lexical_candidate_edges": len(edges),
            "unresolved_call_signatures": len(graph.unresolved),
            "ambiguous_call_signatures": len(graph.ambiguous),
            "rows": rows,
        })
    return {
        "research_question": "Does optional failure-conditioned dependency expansion protect useful failure evidence without the cost of forced closure?",
        "methods": list(POLICIES),
        "budgets": list(BUDGETS),
        "scope": "Archival pre-LLM selector comparison, not a localization benchmark",
        "development_bugs": ["Lang-1", "Time-1"],
        "evaluation_bugs": ["Time-25"],
        "evaluation_is_independent_project": False,
        "policy_constants_frozen": True,
        "gold_used": False,
        "llm_inference_performed": False,
        "budget_unit": "precomputed L1 sketch token proxy",
        "limitations": [
            "Only 3 previously archived bugs, including two Joda Time instances",
            "Time-25 is separate bug but not separate project or guaranteed unseen",
            "Failure anchors are test-name matching, not gold faulty methods",
            "Dependencies are lexical candidate edges, not Java compiler verified",
            "No model input tokens, LLM predictions, patch generation, or ground-truth localization",
        ],
        "cases": cases,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=Path("results/aitech/failure_conditioned_three_bug.json"))
    args = p.parse_args()
    result = experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(case["bug"], case["split"], "edges", case["lexical_candidate_edges"])
        for row in case["rows"]:
            print(row["budget_proxy"], row["method"],
                  "anchors", row["failure_anchors_kept"],
                  "cost", row["used_proxy_tokens"],
                  "missing", row["unresolved_lexical_edges"])
    print("Saved", args.output)


if __name__ == "__main__":
    main()
