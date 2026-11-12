"""Failure-grounded retrieval with explicit related-API class expansion.

Problem observed in Compress-1: the stack trace identifies a reader
(CpioArchiveInputStream), but the changed class is a paired writer
(CpioArchiveOutputStream). Both exist in the frozen 25-method source
pool, yet the previous eight-method shortlist omitted the writer.

This module adds *candidate retrieval recall*, NOT proof of root cause:
1. protect top two original failure-grounded methods;
2. retrieve source-verified (present in pool) reciprocal API class
   partners InputStream/OutputStream, Reader/Writer, Encoder/Decoder;
3. include a method for each test-named production class when present;
4. fill remaining positions with class-diverse candidates;
5. abstain when a related class is absent instead of inventing code.

All links are lexical API-name heuristics, not compiler call edges.
No patched files, changed-class labels, or gold are consulted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .failure_grounded_rerank import (
    FRAME, failing_test_class_hints, rerank
)


PAIRS = (
    ("InputStream", "OutputStream"),
    ("Reader", "Writer"),
    ("Encoder", "Decoder"),
    ("Parser", "Printer"),
)


@dataclass(frozen=True)
class RelatedRetrieval:
    original_indices: tuple[int, ...]
    related_class_proposals: tuple[tuple[str, str], ...]
    test_name_class_hints: tuple[str, ...]
    binding_verified: bool = False


def owner_of(method_id: str) -> str:
    qualified = method_id.split("(", 1)[0]
    return qualified.rsplit(".", 1)[0]


def choose_behavior_method(items: list[dict], indices: list[int]) -> int:
    """Prefer an implementation method to a constructor in paired APIs.

    Only uses candidate names and *buggy* source snippets. Does not
    assert that the chosen method is implicated by the failing test.
    """
    if not indices:
        raise ValueError("cannot choose from empty source candidates")
    behavior = [
        i for i in indices
        if items[i]["function_id"].split("(", 1)[0].rsplit(".", 1)[-1]
        != owner_of(items[i]["function_id"]).rsplit(".", 1)[-1]
    ]
    if behavior:
        return behavior[0]
    return indices[0]


def sibling_class(class_name: str) -> str | None:
    for suffix_a, suffix_b in PAIRS:
        if class_name.endswith(suffix_a):
            return class_name[:-len(suffix_a)] + suffix_b
        if class_name.endswith(suffix_b):
            return class_name[:-len(suffix_b)] + suffix_a
    return None


def related_shortlist(snapshot: dict, *, count: int = 8,
                      per_class_limit: int = 2) -> RelatedRetrieval:
    items = snapshot["candidates"]
    if not 1 <= count <= len(items):
        raise ValueError("invalid shortlist size")
    if per_class_limit < 1:
        raise ValueError("invalid per-class limit")
    ranked = rerank(snapshot)
    classes: dict[str, list[int]] = {}
    for i in ranked:
        classes.setdefault(owner_of(items[i]["function_id"]), []).append(i)
    selected: list[int] = []
    used: set[int] = set()
    counts: Counter[str] = Counter()

    def add(index: int) -> bool:
        if len(selected) >= count or index in used:
            return False
        owner = owner_of(items[index]["function_id"])
        if counts[owner] >= per_class_limit:
            return False
        selected.append(index)
        used.add(index)
        counts[owner] += 1
        return True

    # Preserve strong first-page failure matches so a noisy inferred
    # relation cannot displace all direct evidence.
    for i in ranked[:min(2, count)]:
        add(i)

    frame_owners = {
        c.replace("$", ".") for c, _ in FRAME.findall(snapshot["context"])
    }
    # Extend direct provenance to API peers ONLY when the actual buggy
    # source pool contains that fully qualified peer class.
    proposals = []
    for source_owner in sorted(frame_owners.intersection(classes)):
        partner = sibling_class(source_owner)
        if partner and partner in classes and len(selected) < count:
            if add(choose_behavior_method(items, classes[partner])):
                proposals.append((source_owner, partner))

    # Test-to-production attribution is a weaker named-class hint.
    test_hints = failing_test_class_hints(snapshot["context"])
    for short in sorted(test_hints):
        matches = [
            full for full in classes
            if full.rsplit(".", 1)[-1] == short
        ]
        for full in sorted(matches):
            if len(selected) >= count:
                break
            if not any(owner_of(items[x]["function_id"]) == full
                       for x in selected):
                add(classes[full][0])

    # Fill remaining capacity from frozen relevance order, favoring
    # source-class diversity. No additional repo lookup is performed.
    for i in ranked:
        if len(selected) >= count:
            break
        add(i)
    # A very small source pool may not have enough classes under the
    # diversity cap. Fall back to existing candidates only.
    for i in ranked:
        if len(selected) >= count:
            break
        if i not in used:
            selected.append(i)
            used.add(i)
    return RelatedRetrieval(
        tuple(selected),
        tuple(proposals),
        tuple(sorted(test_hints)),
        False,
    )
