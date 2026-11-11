"""Budget-safe anchor-first evidence selection for AITECH.

This is a deterministic CPU baseline, not a learned policy. It prioritizes
methods mentioned in failing-test evidence before spending remaining budget
on optional source-level call dependencies. Unfitted dependencies are recorded,
NOT silently described as complete. Uses existing method L1 sketch token costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AnchorSelection:
    selected: tuple[int, ...]
    mandatory_anchors_kept: tuple[int, ...]
    unsatisfied_edges: tuple[tuple[int, int], ...]
    tokens: int


def disambiguate_lexical_edges(
    function_ids: list[str], edges: Iterable[tuple[int, int]]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Reject edges whose callee name is overloaded in a defining class.

    This reduces, but does not eliminate, wrong lexical edges. Real Java
    method binding and overload resolution still require AST/type analysis.
    """
    names = []
    for function_id in function_ids:
        owner, method_sig = function_id.rsplit(".", 1)
        names.append((owner, method_sig.split("(", 1)[0]))
    frequency = {name: names.count(name) for name in set(names)}
    clean, rejected = set(), set()
    for source, dest in edges:
        if not (0 <= source < len(function_ids) and 0 <= dest < len(function_ids)):
            raise ValueError("edge endpoint outside candidate list")
        (clean if frequency[names[dest]] == 1 else rejected).add((source, dest))
    return clean, rejected


def anchor_first_selection(
    costs: list[int],
    anchors: Iterable[int],
    edges: Iterable[tuple[int, int]],
    budget: int,
) -> AnchorSelection:
    """Protect rank-ordered failure anchors, then add optional support.

    Anchors are packed individually in ranking order. Supports are added
    when affordable; selected dependent methods with missing support are
    explicitly reported rather than falsely called dependency-complete.
    Non-anchor methods are then added by rank, provided they fit.
    """
    if budget < 0 or any(not isinstance(c, int) or c <= 0 for c in costs):
        raise ValueError("budget must be nonnegative and costs positive integers")
    n = len(costs)
    anchor_order = list(dict.fromkeys(anchors))
    if any(i < 0 or i >= n for i in anchor_order):
        raise ValueError("anchor index outside candidate list")
    edge_set = set(edges)
    if any(a < 0 or b < 0 or a >= n or b >= n for a, b in edge_set):
        raise ValueError("edge endpoint outside candidate list")
    picked: set[int] = set()
    used = 0

    def add(index: int) -> None:
        nonlocal used
        if index not in picked and used + costs[index] <= budget:
            picked.add(index)
            used += costs[index]

    for anchor in anchor_order:
        add(anchor)
    # Fill required supports in ranked order where possible, without
    # displacing the protected failure anchors. Transitive expansion is
    # bounded by the number of candidates.
    for _ in range(n):
        before = len(picked)
        for caller, callee in sorted(edge_set):
            if caller in picked and callee not in picked:
                add(callee)
        if len(picked) == before:
            break
    for index in range(n):
        add(index)
    missing = tuple(sorted((a, b) for a, b in edge_set if a in picked and b not in picked))
    return AnchorSelection(
        selected=tuple(sorted(picked)),
        mandatory_anchors_kept=tuple(a for a in anchor_order if a in picked),
        unsatisfied_edges=missing,
        tokens=used,
    )
