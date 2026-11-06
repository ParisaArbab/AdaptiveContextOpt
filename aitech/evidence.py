"""Dependency-aware selection of debugging evidence.

The MVP uses only Python's standard library, and does not claim to be a
trained model. An optional learned scorer can later supply relevance scores.
"""
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    tokens: int
    relevance: float = 0.0


@dataclass(frozen=True)
class Selection:
    ids: tuple[str, ...]
    tokens: int
    total_relevance: float
    dangling: tuple[str, ...]


def missing_dependencies(selected: Iterable[str], dependencies: Mapping[str, Iterable[str]]) -> tuple[str, ...]:
    """Return selected IDs with at least one required direct dependency missing."""
    chosen = set(selected)
    return tuple(sorted(item for item in chosen if any(parent not in chosen for parent in dependencies.get(item, ()))))


def dependency_closure(ids: Iterable[str], dependencies: Mapping[str, Iterable[str]]) -> set[str]:
    """Transitive closure of required support IDs; rejects dependency cycles."""
    state: dict[str, int] = {}
    result: set[str] = set()

    def visit(item: str) -> None:
        if state.get(item) == 1:
            raise ValueError(f"Dependency cycle involving {item}")
        if state.get(item) == 2:
            return
        state[item] = 1
        for parent in dependencies.get(item, ()):
            visit(parent)
        state[item] = 2
        result.add(item)

    for item in ids:
        visit(item)
    return result


def select_evidence(
    evidence: Iterable[Evidence],
    dependencies: Mapping[str, Iterable[str]],
    budget: int,
    *,
    preserve_dependencies: bool = True,
) -> Selection:
    """Greedy utility-per-token selection, optionally preserving dependency closure.

    Budget is measured in caller-provided token counts; this is a baseline,
    not an optimal knapsack solver. Missing dependency IDs are rejected.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    items = list(evidence)
    by_id = {item.id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("duplicate evidence IDs")
    if any(item.tokens <= 0 for item in items):
        raise ValueError("each evidence item must have positive token cost")
    for item, parents in dependencies.items():
        if item in by_id:
            for parent in parents:
                if parent not in by_id:
                    raise ValueError(f"unknown dependency {parent} for {item}")
    chosen: set[str] = set()
    while True:
        best = None
        for item in items:
            if item.id in chosen:
                continue
            group = (dependency_closure((item.id,), dependencies) if preserve_dependencies
                     else {item.id}) - chosen
            if any(parent not in by_id for parent in group):
                raise ValueError(f"unknown required evidence in {item.id}")
            cost = sum(by_id[x].tokens for x in group)
            if cost == 0 or sum(by_id[x].tokens for x in chosen) + cost > budget:
                continue
            gain = sum(max(0.0, by_id[x].relevance) for x in group)
            candidate = (gain / cost, gain, item.id, group)
            if best is None or candidate[:2] > best[:2] or (
                candidate[:2] == best[:2] and candidate[2] < best[2]
            ):
                best = candidate
        if best is None or best[1] <= 0:
            break
        chosen.update(best[3])
    ordered = tuple(item.id for item in items if item.id in chosen)
    return Selection(
        ids=ordered,
        tokens=sum(by_id[x].tokens for x in ordered),
        total_relevance=sum(by_id[x].relevance for x in ordered),
        dangling=missing_dependencies(ordered, dependencies),
    )
