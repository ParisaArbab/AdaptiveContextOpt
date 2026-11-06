"""Metrics for comparing evidence selectors without using gold labels at inference."""
from dataclasses import dataclass
from typing import Iterable, Mapping
from .evidence import Selection, missing_dependencies


@dataclass(frozen=True)
class Metrics:
    kept_tokens: int
    compression_ratio: float
    dependency_violation_count: int
    gold_evidence_recall: float | None


def evaluate(
    selected: Selection,
    *,
    raw_tokens: int,
    dependencies: Mapping[str, Iterable[str]],
    gold_evidence: Iterable[str] | None = None,
) -> Metrics:
    """Evaluate a selection. Gold IDs are used ONLY in offline evaluation."""
    if raw_tokens <= 0:
        raise ValueError("raw_tokens must be positive")
    if selected.tokens > raw_tokens:
        raise ValueError("selection exceeds raw token count")
    gold = None if gold_evidence is None else set(gold_evidence)
    recall = None if gold is None or not gold else len(set(selected.ids) & gold) / len(gold)
    return Metrics(
        kept_tokens=selected.tokens,
        compression_ratio=selected.tokens / raw_tokens,
        dependency_violation_count=len(missing_dependencies(selected.ids, dependencies)),
        gold_evidence_recall=recall,
    )
