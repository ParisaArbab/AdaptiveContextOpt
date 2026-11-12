"""Failure-conditioned marginal-utility evidence selection.

Research hypothesis: under a fixed context budget, preserving test/trace
anchors before selectively buying dependency support should avoid the
high-cost full-closure failure mode. This is a deterministic pre-LLM
selector, NOT a trained ML method or established research novelty.

All features come from buggy snapshot and failing test before gold scoring.
Dependency hints must be explicitly marked as lexical or compiler-verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Selection:
    selected: tuple[int, ...]
    proxy_tokens: int
    anchors_kept: int
    anchors_total: int
    unresolved_edges: int
    evidence_score: float
    ids: tuple[str, ...]


def failure_anchors(items: list[dict], failure_text: str) -> tuple[int, ...]:
    """Detect names explicitly mentioned as calls in failing evidence.

    Only a weak heuristic, especially for overloaded constructors.
    """
    return tuple(
        i for i, item in enumerate(items)
        if item["function_id"].split("(", 1)[0].rsplit(".", 1)[-1] + "(" in failure_text
    )


def relevance(item: dict, rank: int) -> float:
    """Fixed formula, no gold labels or fitted bug-specific coefficients."""
    f = item.get("features", {})
    prior = min(1.0, max(0.0, float(item.get("prior", f.get("prior", 0)))))
    trace = min(1.0, max(0.0, float(f.get("trace_match", 0))))
    failure = min(1.0, max(0.0, float(f.get("failure_relevance", 0))))
    agreement = min(1.0, max(0.0, float(f.get("ranker_agreement", 0))))
    return (
        0.35 * prior + 0.35 * failure +
        0.20 * trace + 0.10 * agreement +
        0.10 / (rank + 1)
    )


def select(
    items: list[dict],
    failure_text: str,
    budget: int,
    *,
    method: str,
    edges: Iterable[tuple[int, int]] = (),
    graph_confidence: str = "lexical",
) -> Selection:
    """Pack whole L1 method sketches, with optional conditional graph bonus.

    Supported arms: rank_fill, anchor_first, relevance_density,
    failure_graph. 'failure_graph' buys an optional dependency only
    when it offers a good marginal score per proxy token; no forced closure.
    Graph confidence must be declared, to avoid calling lexical hints verified.
    """
    allowed = {"rank_fill", "anchor_first", "relevance_density", "failure_graph"}
    if method not in allowed:
        raise ValueError("unknown policy")
    if graph_confidence not in {"lexical", "compiler"}:
        raise ValueError("graph provenance must be lexical or compiler")
    if budget < 0:
        raise ValueError("negative budget")
    n = len(items)
    costs = [int(item["features"]["l1_tokens"]) for item in items]
    if any(x <= 0 for x in costs):
        raise ValueError("invalid sketch token estimate")
    graph = set(edges)
    if any(not (0 <= a < n and 0 <= b < n) for a, b in graph):
        raise ValueError("edge refers to missing candidate")

    anchors = failure_anchors(items, failure_text)
    chosen: set[int] = set()
    remaining = budget

    def add(i: int) -> bool:
        nonlocal remaining
        if i in chosen or costs[i] > remaining:
            return False
        chosen.add(i)
        remaining -= costs[i]
        return True

    if method == "rank_fill":
        for i in range(n):
            add(i)
    else:
        # Always give failure-observed evidence first refusal of budget.
        for i in anchors:
            add(i)
        if method == "anchor_first":
            for i in range(n):
                add(i)
        else:
            base = [relevance(item, i) for i, item in enumerate(items)]
            while True:
                scores = []
                for i in range(n):
                    if i in chosen or costs[i] > remaining:
                        continue
                    # Graph links are optional, not all-or-nothing.
                    # Compiler-bound edges earn more than lexical hints.
                    support = sum(1 for a, b in graph if a in chosen and b == i)
                    bonus = 0.0
                    if method == "failure_graph" and support:
                        bonus = (0.30 if graph_confidence == "compiler" else 0.12) * min(support, 2)
                    value = base[i] + bonus
                    scores.append((value / (costs[i] ** 0.70), -i, i))
                if not scores:
                    break
                best = max(scores)[2]
                add(best)

    ids = tuple(items[i]["function_id"] for i in sorted(chosen))
    unresolved = sum(1 for a, b in graph if a in chosen and b not in chosen)
    return Selection(
        selected=tuple(sorted(chosen)),
        proxy_tokens=budget - remaining,
        anchors_kept=len(chosen.intersection(anchors)),
        anchors_total=len(anchors),
        unresolved_edges=unresolved,
        evidence_score=round(sum(relevance(items[i], i) for i in chosen), 6),
        ids=ids,
    )
