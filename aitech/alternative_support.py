"""Alternative support-path selection for debugging evidence.

Referential completeness is not the same as preserving EVERY graph edge.
For a retained evidence item, one of several sufficient support bundles
may be enough. These bundles are OR-of-AND requirements:

  support["failed_test"] = (("call_bridge",), ("definition", "assertion"))

One entire bundle must be retained, but not all bundles. This module
implements a small exact budget solver for <= 18 evidence items.

WARNING: Support bundles must be justified by source provenance or
manually annotated ground truth. This code does not infer causal
dependencies, does not reproduce a trained detector from the paper,
and does not measure downstream LLM localization.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from .evidence import Evidence

SupportMap = Mapping[str, Sequence[Sequence[str]]]


@dataclass(frozen=True)
class AlternativeSelection:
    ids: tuple[str, ...]
    tokens: int
    relevance: float
    supported_ids: tuple[str, ...]
    dangling_ids: tuple[str, ...]
    evaluated_subsets: int


def dangling_evidence(
    selected: Sequence[str], support_sets: SupportMap
) -> tuple[str, ...]:
    """Return items lacking a grounded, fully retained support alternative.

    Derive support from unconstrained selected evidence upward. Requiring
    support paths to terminate at such evidence prevents purely circular
    claims (a supports b and b supports a) from counting as complete.
    """
    chosen = set(selected)
    supported = {item for item in chosen if item not in support_sets}
    for _ in range(len(chosen)):
        before = len(supported)
        for item in sorted(chosen - supported):
            if any(set(option).issubset(supported)
                   for option in support_sets[item]):
                supported.add(item)
        if len(supported) == before:
            break
    return tuple(sorted(chosen - supported))


def _validate(items: list[Evidence], support_sets: SupportMap, budget: int) -> None:
    if budget < 0:
        raise ValueError("negative budget")
    ids = {item.id for item in items}
    if len(ids) != len(items):
        raise ValueError("duplicate evidence IDs")
    if len(items) > 18:
        raise ValueError("exact search only supports up to 18 evidence items")
    if any(item.tokens <= 0 or not isfinite(item.relevance) for item in items):
        raise ValueError("invalid cost or relevance")
    for item, options in support_sets.items():
        if item not in ids:
            raise ValueError(f"unknown evidence item: {item}")
        if not options:
            raise ValueError(f"no support alternatives for {item}")
        for option in options:
            if not option:
                raise ValueError(f"empty support bundle for {item}")
            if len(set(option)) != len(option):
                raise ValueError(f"duplicate support in {item}")
            if item in option:
                raise ValueError(f"self support in {item}")
            if not set(option).issubset(ids):
                raise ValueError(f"unknown support in {item}")


def exact_select(
    evidence: Sequence[Evidence],
    support_sets: SupportMap,
    budget: int,
    *,
    require_complete: bool = True,
) -> AlternativeSelection:
    """Maximize retained relevance at fixed budget, with an honest exact oracle.

    Enumerates all subsets, so it is a small-case *upper-bound reference*,
    not a scalable adaptive retrieval method. Complete solutions must
    satisfy one support option per selected constrained item; nested support
    requirements are checked across the entire selected subset.

    Tie breaking is deterministic: best relevance, then fewer tokens,
    then lexicographic ordered IDs.
    """
    items = list(evidence)
    _validate(items, support_sets, budget)
    n = len(items)
    candidates = 0
    best_key = None
    best = None
    for mask in range(1 << n):
        candidates += 1
        chosen = tuple(item.id for i, item in enumerate(items) if mask & (1 << i))
        cost = sum(item.tokens for i, item in enumerate(items) if mask & (1 << i))
        if cost > budget:
            continue
        dangling = dangling_evidence(chosen, support_sets)
        if require_complete and dangling:
            continue
        score = sum(item.relevance for i, item in enumerate(items) if mask & (1 << i))
        key = (-score, cost, chosen)
        if best_key is None or key < best_key:
            best_key = key
            best = (chosen, cost, score, dangling)
    if best is None:
        raise AssertionError("empty selection must always fit")
    chosen, cost, score, dangling = best
    return AlternativeSelection(
        ids=chosen,
        tokens=cost,
        relevance=round(score, 9),
        supported_ids=tuple(i for i in chosen if i in support_sets and i not in dangling),
        dangling_ids=dangling,
        evaluated_subsets=candidates,
    )
