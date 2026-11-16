"""Regression tests for duplicate-ID BM25/lexical candidate recovery."""
from types import SimpleNamespace

import pytest

from rave_fl.eval import candidates
from rave_fl.eval.schema import CodeEntity


def entity(name, source="relevant signal"):
    return CodeEntity(
        entity_id=name, path="src/Main.java", start_line=1, end_line=2,
        source=source, prior=0.0, kind="method",
    )


@pytest.mark.parametrize("provider", ["bm25", "lexical"])
def test_distinct_method_ids_fill_quota(monkeypatch, provider):
    # Original top-24 contains duplicates. At least 24 distinct methods exist.
    pool = [entity("src/Main.java::same", "relevant signal")] * 4
    pool += [entity(f"src/Main.java::method{i:02d}", "relevant signal") for i in range(30)]
    monkeypatch.setattr(candidates, "index_source", lambda *_: pool)
    instance = SimpleNamespace(
        issue="relevant signal",
        evidence=SimpleNamespace(text="relevant signal"),
    )
    opts = {
        "candidate_provider": provider, "candidate_count": 24,
        "granularity": "method", "max_source_files": 3000,
        "max_source_bytes": 30000000,
    }
    first, provenance = candidates.provide(instance, opts)
    second, _ = candidates.provide(instance, opts)
    ids = [e.entity_id for e in first]
    assert len(ids) == len(set(ids)) == 24
    assert ids == [e.entity_id for e in second]  # deterministic
    assert provenance["retrieved_unique_entities"] == 24
    assert provenance["indexed_entities"] == 34
    assert ids[0] == "src/Main.java::method00" or ids[0] == "src/Main.java::same"


def test_insufficient_distinct_methods_remains_insufficient(monkeypatch):
    pool = [entity("src/Main.java::duplicate")] * 50
    monkeypatch.setattr(candidates, "index_source", lambda *_: pool)
    instance = SimpleNamespace(
        issue="relevant", evidence=SimpleNamespace(text="relevant"),
    )
    opts = {"candidate_provider": "bm25", "candidate_count": 24,
            "granularity": "method", "max_source_files": 3000,
            "max_source_bytes": 30000000}
    selected, provenance = candidates.provide(instance, opts)
    assert len(selected) == 1
    assert provenance["retrieved_unique_entities"] == 1


def test_best_scoring_duplicate_is_retained(monkeypatch):
    pool = [entity("src/Main.java::same", "noise noise"),
            entity("src/Main.java::same", "important important important"),
            entity("src/Main.java::other", "important")]
    monkeypatch.setattr(candidates, "index_source", lambda *_: pool)
    instance = SimpleNamespace(
        issue="important", evidence=SimpleNamespace(text="important"),
    )
    opts = {"candidate_provider": "bm25", "candidate_count": 2,
            "granularity": "method", "max_source_files": 3000,
            "max_source_bytes": 30000000}
    selected, _ = candidates.provide(instance, opts)
    assert len(selected) == 2
    retained = next(e for e in selected if e.entity_id == "src/Main.java::same")
    assert retained.source == "important important important"
