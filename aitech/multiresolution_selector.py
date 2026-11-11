"""Failure-conditioned multi-resolution evidence allocation.

Pilot idea: reserve low-cost L0 signatures for failure-name anchors, then
adaptively upgrade to L1 structural method excerpts based on marginal
relevance per added token.

Novelty is a HYPOTHESIS, not established. L0 identity retention is not
equivalent to retaining executable code. No gold labels, training, LLM
inference, or true tokenized prompt costs are used in this selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .failure_conditioned_selector import failure_anchors, relevance


@dataclass(frozen=True)
class TieredSelection:
    levels: tuple[int, ...]  # 0=not selected, 1=L0 identity, 2=L1 structural
    proxy_tokens: int
    anchor_identities_retained: int
    anchors_total: int
    anchor_l1_retained: int
    l0_count: int
    l1_count: int
    weighted_proxy_utility: float
    unresolved_edges: int


def select_tiered(
    items: list[dict],
    failure_text: str,
    budget: int,
    *,
    edges: Iterable[tuple[int, int]] = (),
    graph_confidence: str = "lexical",
) -> TieredSelection:
    if budget < 0:
        raise ValueError("negative budget")
    if graph_confidence not in ("lexical", "compiler"):
        raise ValueError("unsupported graph provenance")
    n = len(items)
    l0 = [int(item["features"]["l0_tokens"]) for item in items]
    l1 = [int(item["features"]["l1_tokens"]) for item in items]
    if any(a <= 0 or b <= 0 for a, b in zip(l0, l1)):
        raise ValueError("invalid tokens")
    # Some stored lossy sketches have a smaller estimated token count than
    # their signature-only version. Do not assign an upgrade negative cost.
    l1 = [max(a, b) for a, b in zip(l0, l1)]
    graph = set(edges)
    if any(not (0 <= a < n and 0 <= b < n) for a, b in graph):
        raise ValueError("invalid graph edge")
    anchors = failure_anchors(items, failure_text)
    scores = [relevance(item, i) for i, item in enumerate(items)]
    levels = [0] * n
    remaining = budget

    def add(i: int, level: int) -> bool:
        nonlocal remaining
        if level <= levels[i]:
            return False
        old = 0 if levels[i] == 0 else l0[i] if levels[i] == 1 else l1[i]
        new = l0[i] if level == 1 else l1[i]
        increment = new - old
        if increment > remaining:
            return False
        levels[i] = level
        remaining -= increment
        return True

    # Prioritize naming the involved methods, without buying expensive
    # complete structural snippets before seeing marginal evidence value.
    for i in anchors:
        add(i, 1)

    while True:
        choices = []
        for i in range(n):
            if levels[i] == 2:
                continue
            next_level = levels[i] + 1
            incremental_cost = l0[i] if next_level == 1 else l1[i] - l0[i]
            if incremental_cost > remaining:
                continue
            # A signature carries identity but not full implementation.
            # Relative values (0.20, 1.00) are fixed proxies, not measured
            # predictive accuracy or model utilities.
            fidelity_gain = 0.20 if next_level == 1 else 0.80
            support = sum(1 for a, b in graph if levels[a] > 0 and b == i)
            provenance_weight = 0.20 if graph_confidence == "compiler" else 0.07
            graph_bonus = provenance_weight * min(support, 2)
            # Failure anchors have a slight upgrade preference but are
            # NOT forced to displace all cheaper useful methods.
            anchor_bonus = 0.08 if i in anchors and next_level == 2 else 0.0
            value = scores[i] * fidelity_gain + graph_bonus + anchor_bonus
            choices.append((value / (max(1, incremental_cost) ** 0.7), -i, i, next_level))
        if not choices:
            break
        choice = max(choices)
        add(choice[2], choice[3])

    return TieredSelection(
        levels=tuple(levels),
        proxy_tokens=budget - remaining,
        anchor_identities_retained=sum(levels[i] > 0 for i in anchors),
        anchors_total=len(anchors),
        anchor_l1_retained=sum(levels[i] == 2 for i in anchors),
        l0_count=sum(v == 1 for v in levels),
        l1_count=sum(v == 2 for v in levels),
        weighted_proxy_utility=round(
            sum(scores[i] * (0.20 if level == 1 else 1.0 if level == 2 else 0.0)
                for i, level in enumerate(levels)), 6
        ),
        unresolved_edges=sum(1 for a, b in graph
                             if levels[a] and levels[b] == 0),
    )
