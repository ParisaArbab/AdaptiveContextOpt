"""Failure-test-name candidate ranking as a separate, auditable heuristic.

Test classes often follow conventions such as FooTest, FooTests,
TestFoo, or a .junit. subpackage. These names can be linked to
production class owners BEFORE viewing a patch, but the link is
heuristic, not compiler-verified or proof of bug causality.

The module augments, and DOES NOT retroactively change, the prior
stack-frame/IDF baseline.
"""
from __future__ import annotations

import re
from .failure_aligned_reranker import failure_rerank

TEST_ID = re.compile(r"([a-zA-Z_$][\w$]*(?:\.[a-zA-Z_$][\w$]*)+)::[\w$]+")
TEST_FRAME = re.compile(r"\bat\s+([\w.$]+)\.(?:test\w*|setUp|tearDown)\(")


def _production_hint(test_class: str) -> tuple[str, str] | None:
    pieces = test_class.replace("$", ".").split(".")
    if not pieces:
        return None
    short = pieces[-1]
    if short.endswith("Tests") and len(short) > 5:
        short = short[:-5]
    elif short.endswith("Test") and len(short) > 4:
        short = short[:-4]
    elif short.startswith("Test") and len(short) > 4:
        short = short[4:]
    else:
        return None
    # Remove only standard Java test-namespace segments; do not guess
    # arbitrary package changes.
    package = [x for x in pieces[:-1] if x.lower() not in ("junit", "tests", "test")]
    return ".".join(package + [short]), short


def test_hints(context: str) -> tuple[tuple[str, str], ...]:
    named = TEST_ID.findall(context)
    named += TEST_FRAME.findall(context)
    hints = set()
    for test_class in named:
        hint = _production_hint(test_class)
        if hint is not None:
            hints.add(hint)
    return tuple(sorted(hints))


def rank_with_test_hints(snapshot: dict) -> list[dict]:
    base = failure_rerank(snapshot)
    hints = test_hints(snapshot["context"])
    scored = []
    for row in base:
        owner = row["method"].split("(", 1)[0].rsplit(".", 1)[0]
        short = owner.rsplit(".", 1)[-1]
        # Same fully-qualified package after removing a conventional
        # .junit. package is stronger than same short class name alone.
        exact = any(owner == qualified for qualified, _ in hints)
        partial = any(short == suffix for _, suffix in hints)
        boost = 30.0 if exact else 12.0 if partial else 0.0
        scored.append({
            **row,
            "test_name_boost": boost,
            "test_name_hint": exact or partial,
            "score": round(row["score"] + boost, 6),
        })
    return sorted(scored, key=lambda row: (-row["score"], row["index"]))


def hinted_shortlist(snapshot: dict, limit: int = 8) -> list[int]:
    if limit < 2:
        raise ValueError("limit must be at least two")
    chosen = []
    per_file: dict[str, int] = {}
    for row in rank_with_test_hints(snapshot):
        path = row["source_path"] or row["method"].rsplit(".", 1)[0]
        if per_file.get(path, 0) >= 2:
            continue
        chosen.append(row["index"])
        per_file[path] = per_file.get(path, 0) + 1
        if len(chosen) == limit:
            break
    return chosen
