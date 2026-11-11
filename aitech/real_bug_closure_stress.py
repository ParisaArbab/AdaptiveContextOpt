"""CPU-only cross-bug dependency closure stress test on archived RAVE-FL inputs.

Use archived independent_live/{Lang-1,Time-1}/input_snapshot.json.
All inputs are pre-existing buggy Java sources, ranked candidates and test
failure text. NO gold patch and NO LLM at inference or evaluation.

IMPORTANT: Same-class, same-name lexical calls are ambiguous for overloaded
methods and can introduce false edges or cycles. This is a deliberately
imperfect stress-test baseline, NOT verified Java call-graph extraction.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("results/rave_fl_iteration_d/independent_live")
BUGS = ("Lang-1", "Time-1")
BUDGETS = (200, 300, 450, 600)


def simple_name(function_id: str) -> str:
    return function_id.rsplit(".", 1)[-1].split("(", 1)[0]


def owner(function_id: str) -> str:
    return function_id.rsplit(".", 1)[0]


def lexical_edges(candidates: list[dict]) -> set[tuple[int, int]]:
    edges = set()
    for i, caller in enumerate(candidates):
        # Short L1 summary contains lossy source calls. This is not AST.
        text = caller["levels"][1]
        for j, callee in enumerate(candidates):
            name = simple_name(callee["function_id"])
            if i != j and len(name) > 4 and owner(caller["function_id"]) == owner(callee["function_id"]):
                if re.search(r"\b" + re.escape(name) + r"\s*\(", text):
                    edges.add((i, j))
    return edges


def closure(start: int, edges: set[tuple[int, int]]) -> set[int]:
    reached, todo = set(), [start]
    while todo:
        current = todo.pop()
        if current in reached:
            continue
        reached.add(current)
        todo.extend(target for src, target in edges if src == current)
    return reached


def run_one(data: dict, bug: str) -> dict:
    candidates = data["candidates"]
    edges = lexical_edges(candidates)
    context = data["context"]
    anchors = {
        i for i, item in enumerate(candidates)
        if simple_name(item["function_id"]) + "(" in context
    }
    costs = [int(item["features"]["l1_tokens"]) for item in candidates]
    rows = []
    for budget in BUDGETS:
        for method in ("rank_fill", "closure", "anchor_reserved"):
            selected: set[int] = set()
            if method == "anchor_reserved":
                for i in sorted(anchors)[:2]:
                    expanded = closure(i, edges)
                    if sum(costs[j] for j in selected | expanded) <= budget:
                        selected.update(expanded)
            for i in range(len(candidates)):
                expanded = {i} if method == "rank_fill" else closure(i, edges)
                if sum(costs[j] for j in selected | expanded) <= budget:
                    selected.update(expanded)
            rows.append({
                "bug": bug,
                "budget": budget,
                "method": method,
                "selected": len(selected),
                "used_proxy_tokens": sum(costs[j] for j in selected),
                "failure_name_anchors_kept": len(selected & anchors),
                "failure_name_anchors_total": len(anchors),
                "known_lexical_dangling": sum(
                    1 for src, dst in edges if src in selected and dst not in selected
                ),
                "selected_ids": [candidates[i]["function_id"] for i in sorted(selected)],
            })
    return {
        "bug": bug,
        "source_candidate_count": len(candidates),
        "failure_name_anchors": [candidates[i]["function_id"] for i in sorted(anchors)],
        "lexical_edges": [
            [candidates[a]["function_id"], candidates[b]["function_id"]]
            for a, b in sorted(edges)
        ],
        "rows": rows,
    }


def experiment(root: Path = ROOT) -> dict:
    results = []
    for bug in BUGS:
        path = root / bug / "input_snapshot.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(run_one(data, bug))
    return {
        "scope": "two archived independent Defects4J inputs, development-only",
        "bugs": list(BUGS),
        "methods": ["rank_fill", "closure", "anchor_reserved"],
        "budget_unit": "precomputed L1 token proxy, not Ollama input tokens",
        "source_path_pattern": str(root / "{bug}" / "input_snapshot.json"),
        "limitations": [
            "Only two bugs, no independent test split",
            "Lexical calls can misresolve overloads and create false cycles",
            "No gold labels, no direct fault-localization or downstream model accuracy",
            "Same-class approximate edges exclude cross-class dependencies",
            "The anchor-reservation approach is naive and may fail if closure cannot fit",
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/aitech/real_bug_closure_stress.json"))
    args = parser.parse_args()
    result = experiment(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for record in result["results"]:
        print(record["bug"], "anchors", len(record["failure_name_anchors"]),
              "edges", len(record["lexical_edges"]))
        for row in record["rows"]:
            print(row["budget"], row["method"],
                  "anchors", str(row["failure_name_anchors_kept"]) + "/" + str(row["failure_name_anchors_total"]),
                  "dangling", row["known_lexical_dangling"],
                  "tokens", row["used_proxy_tokens"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
