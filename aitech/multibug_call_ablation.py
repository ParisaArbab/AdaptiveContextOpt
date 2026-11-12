"""Development CPU study: rank-first vs call-name dependency closure.

Inputs are preexisting Defects4J Java source snapshots pinned in this
repository, not freshly executed test failures. Regex edges are *candidate*
call links, NOT resolved Java call graphs and NOT causality labels.

No patch/gold labels are read. No LLM runs. Output should not be used to
claim improved localization. Run from repository root:
 python -m aitech.multibug_call_ablation --output-dir results/aitech/multibug_call
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCES = {
    "Lang-1": Path("results/rave_fl_iteration_d/independent_live/Lang-1/input_snapshot.json"),
    "Time-25": Path("results/rave_fl_rank_repair/live_original/Time-25/input_snapshot.json"),
}
BUDGET_PCT = (25, 35, 50)
MODES = ("rank_fill", "closure", "protected")


def method_name(function_id: str) -> str:
    return function_id.split("(", 1)[0].rsplit(".", 1)[-1]


def candidate_graph(items: list[dict]) -> dict[int, list[int]]:
    """Conservative-ish lexical call candidates, no dispatch/overload resolution.

    A repeated name can map to the first candidate encountered and is not
    necessarily the true call target. Never interpret the graph as verified.
    """
    graph: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        seen_names: set[str] = set()
        edges = []
        for other, target in enumerate(items):
            if index == other:
                continue
            name = target["name"]
            if not re.fullmatch(r"[A-Za-z_]\w{3,}", name):
                continue
            if name in seen_names:
                continue
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(", item["source"]):
                edges.append(other)
                seen_names.add(name)
        graph[index] = edges
    return graph


def select(items: list[dict], graph: dict[int, list[int]], budget: int, mode: str) -> dict:
    chosen: set[int] = set()
    words = 0

    def closure(start: int) -> set[int]:
        pending = [start]
        group: set[int] = set()
        while pending:
            i = pending.pop()
            if i in group or i in chosen:
                continue
            group.add(i)
            pending.extend(graph[i])
        return group

    for index, item in enumerate(items):
        if index in chosen:
            continue
        group = {index} if mode == "rank_fill" else closure(index)
        size = sum(items[k]["words"] for k in group)
        if words + size <= budget:
            chosen.update(group)
            words += size
        elif mode == "protected" and words + item["words"] <= budget:
            chosen.add(index)
            words += item["words"]

    retained_edges = [(i, j) for i in range(len(items))
                      if i in chosen for j in graph[i]]
    closed_edges = sum(1 for _, target in retained_edges if target in chosen)
    return {
        "method": mode,
        "words": words,
        "kept": len(chosen),
        "rank1_selected": 0 in chosen,
        "retained_edges": len(retained_edges),
        "closed_edges": closed_edges,
        "edge_completion": (closed_edges / len(retained_edges)
                            if retained_edges else None),
        "selected_ids": [items[i]["id"] for i in sorted(chosen)],
    }


def run_snapshots(sources: dict[str, Path] = SOURCES) -> dict:
    rows = []
    provenance = {}
    for bug, path in sources.items():
        raw = path.read_bytes()
        snapshot = json.loads(raw)
        items = []
        for i, record in enumerate(snapshot["candidates"]):
            src = record["source"]
            items.append({"id": record["function_id"], "source": src,
                          "name": method_name(record["function_id"]),
                          "words": len(src.split())})
        graph = candidate_graph(items)
        total = sum(x["words"] for x in items)
        provenance[bug] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_count": len(items),
            "total_words": total,
            "lexical_edges": sum(map(len, graph.values())),
        }
        for pct in BUDGET_PCT:
            budget = total * pct // 100
            for mode in MODES:
                row = select(items, graph, budget, mode)
                rows.append({"bug": bug, "budget_pct": pct, "budget": budget, **row})
    return {
        "scope": "Development-only archived Defects4J candidate-source ablation",
        "source": "Two previously archived repository snapshots, no fresh test execution",
        "independent_bugs": len(sources),
        "gold_labels_used": False,
        "graph_quality": "lexical call-name candidate edges, not verified call graph",
        "method": "whole-method word budgets and rank-order greedy selection",
        "not_measured": ["fault localization accuracy", "patch correctness",
                         "tokenized LLM input", "LLM cost", "end-to-end inference"],
        "provenance": provenance, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/aitech/multibug_call"))
    args = parser.parse_args()
    result = run_snapshots()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(result['rows'])} rows to {output}")
    for r in result["rows"]:
        print(f"{r['bug']:7} {r['budget_pct']:2}% {r['method']:10} "
              f"{r['words']:5} words {r['kept']:2} methods "
              f"edge completion={r['edge_completion']}")


if __name__ == "__main__":
    main()
