"""Matched-budget selection study driven by compiler-mapped candidate graph.

All evidence candidates are frozen before any gold-label evaluation.
No trained policy or LLM is invoked. This only tests evidence packing.
"""
from __future__ import annotations

from .anchor_budget import anchor_first_selection


def compare_policies(
    candidate_ids: list[str],
    costs: list[int],
    failing_context: str,
    edges: list[tuple[int, int]],
    budgets: tuple[int, ...] = (200, 300, 450, 600),
) -> list[dict]:
    if len(costs) != len(candidate_ids):
        raise ValueError("cost and candidate lengths disagree")
    if any(cost <= 0 for cost in costs):
        raise ValueError("nonpositive evidence cost")
    # Failure references are determined solely from failing-test text,
    # not patch labels. Name matches are a weak proxy for relevant evidence.
    def name(function_id: str) -> str:
        return function_id.split("(", 1)[0].rsplit(".", 1)[-1]
    anchors = [i for i, identifier in enumerate(candidate_ids)
               if name(identifier) + "(" in failing_context]
    rows = []
    for budget in budgets:
        if budget < 0:
            raise ValueError("negative token budget")
        for method in ("rank_fill", "anchor_first", "compiler_graph_first"):
            if method == "rank_fill":
                selected = set()
                used = 0
                for i, cost in enumerate(costs):
                    if used + cost <= budget:
                        selected.add(i)
                        used += cost
            else:
                result = anchor_first_selection(
                    costs, anchors,
                    edges if method == "compiler_graph_first" else [],
                    budget,
                )
                selected = set(result.selected)
                used = result.tokens
            dangling = sum(1 for src, dst in edges
                           if src in selected and dst not in selected)
            rows.append({
                "method": method,
                "budget_proxy": budget,
                "selected_proxy_tokens": used,
                "failure_name_anchors_kept": len(selected.intersection(anchors)),
                "failure_name_anchors_total": len(anchors),
                "unresolved_mapped_support": dangling,
                "selected_ids": [candidate_ids[i] for i in sorted(selected)],
            })
    return rows
