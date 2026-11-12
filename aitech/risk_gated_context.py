"""Risk-gated dependency expansion with protected failure anchors.

Research hypothesis: optional code graph expansion should be justified by
marginal utility per token, not forced even when it displaces stronger
ranked evidence. No gold patches, learned policy or LLM required.

Confidence values are INPUT ESTIMATES. Confidence calibration and true
localization effectiveness are not established by this selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Edge:
    caller: int
    callee: int
    confidence: float
    provenance: str


@dataclass(frozen=True)
class Decision:
    selected: tuple[int, ...]
    tokens: int
    anchors_retained: tuple[int, ...]
    unsatisfied: tuple[tuple[int, int], ...]
    graph_accepted: tuple[tuple[int, int], ...]
    graph_rejected: tuple[tuple[int, int], ...]
    decisions: tuple[dict, ...]


def select_context(
    costs: list[int], anchors: Iterable[int], edges: Iterable[Edge],
    budget: int, *, graph_weight: float = 1.,
    min_confidence: float = .75, rank_weight: float = 1.,
) -> Decision:
    """Pick failure anchors first, then select highest marginal gain/token.

    For candidate i, utility is relevance + support, divided by its cost:
      relevance = rank_weight/(rank(i)+1)
      support = graph_weight*sum(confidence of incoming eligible edges
                                 from already selected callers)
    Each iteration recomputes gain given selected evidence. Missing graph
    support is recorded rather than described as satisfied.
    """
    n = len(costs)
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be nonnegative integer")
    if any(type(c) is not int or c <= 0 for c in costs):
        raise ValueError("costs must be positive integers")
    if not 0 <= min_confidence <= 1 or graph_weight < 0 or rank_weight <= 0:
        raise ValueError("invalid confidence or weights")
    anchors_order = tuple(dict.fromkeys(anchors))
    if any(type(i) is not int or not 0 <= i < n for i in anchors_order):
        raise ValueError("anchor outside candidate list")
    edge_list = tuple(edges)
    if any(not (0 <= e.caller < n and 0 <= e.callee < n and
                0 <= e.confidence <= 1 and bool(e.provenance)) for e in edge_list):
        raise ValueError("invalid graph edge")
    selected: set[int] = set()
    used = 0
    decisions: list[dict] = []
    for anchor in anchors_order:
        if anchor not in selected and used + costs[anchor] <= budget:
            selected.add(anchor)
            used += costs[anchor]
            decisions.append({"id": anchor, "reason": "failure_anchor",
                              "gain_per_token": None})
    eligible = tuple(e for e in edge_list if e.confidence >= min_confidence)
    while True:
        feasible = []
        for i, cost in enumerate(costs):
            if i in selected or used + cost > budget:
                continue
            support = sum(e.confidence * graph_weight for e in eligible
                          if e.callee == i and e.caller in selected)
            gain = (rank_weight/(i+1) + support)/cost
            feasible.append((gain, i, support))
        if not feasible:
            break
        gain, i, support = max(feasible, key=lambda row: (row[0], -row[1]))
        selected.add(i)
        used += costs[i]
        decisions.append({
            "id": i, "reason": "graph_support" if support > 0 else "rank",
            "gain_per_token": round(gain, 12),
            "support_gain": round(support, 12),
        })
    return Decision(
        tuple(sorted(selected)), used,
        tuple(i for i in anchors_order if i in selected),
        tuple(sorted({(e.caller, e.callee) for e in eligible
                      if e.caller in selected and e.callee not in selected})),
        tuple(sorted({(e.caller, e.callee) for e in eligible
                      if e.caller in selected and e.callee in selected})),
        tuple(sorted({(e.caller, e.callee) for e in edge_list
                      if e.confidence < min_confidence})),
        tuple(decisions),
    )
