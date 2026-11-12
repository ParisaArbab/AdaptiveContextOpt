"""Failure-aligned candidate reranking without gold labels.

Two evidence signals:
1) stack-frame agreement with Java owner and method names;
2) inverse-document-frequency lexical overlap of failed-test text
   with a method's identity and short source sketch.

This is deliberately deterministic, CPU-only and pre-gold. It does
not claim to be FaR-Loc embeddings, a learned model, or true causality.
A trace frame may point to an innocent caller, so use a graded boost,
not an absolute rule. No fixed patch labels or class names appear here.
"""
from __future__ import annotations

from collections import Counter
from math import log
import re

from .live_defects4j_snapshot import FRAME

WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def words(text: str) -> set[str]:
    # Separate Java camel-case and snake-case parts before filtering.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    return {x.lower() for x in WORD.findall(text) if len(x) >= 3}


def failure_rerank(snapshot: dict) -> list[dict]:
    """Return candidate metadata, ranked entirely using buggy-state context."""
    context = snapshot["context"]
    candidates = snapshot["candidates"]
    if not candidates:
        raise ValueError("candidate pool is empty")
    failure_terms = words(context)
    frame_refs = FRAME.findall(context)
    # Stack depth encodes proximity to the observed failure, but may be
    # dominated by test framework calls; scores remain only candidates.
    frames = []
    for depth, (owner, method, _, _) in enumerate(frame_refs):
        frames.append((owner.replace("$", "."), method, depth))
    # IDF over *candidate pool*, never repository patches or gold methods.
    docs = [words(c["function_id"] + " " + c["levels"][1][:900])
            for c in candidates]
    freq = Counter(term for doc in docs for term in doc)
    number = len(docs)
    results = []
    for i, (candidate, doc) in enumerate(zip(candidates, docs)):
        method_id = candidate["function_id"]
        owner_method = method_id.split("(", 1)[0]
        owner, method = owner_method.rsplit(".", 1)
        lexical = sum(
            log(1 + (number + 1) / (freq[t] + 1))
            for t in failure_terms & doc
        )
        exact_frame = any(o == owner and m == method for o, m, _ in frames)
        owner_frame = any(o == owner for o, _, _ in frames)
        method_frame = any(m == method for _, m, _ in frames)
        # A matching owner without a matching method is weaker than
        # the exact method call. Class-only alignment can still help.
        trace = (16.0 if exact_frame else 6.0 if owner_frame else
                 1.5 if method_frame else 0.0)
        score = lexical + trace
        results.append({
            "index": i, "method": method_id,
            "score": round(score, 6),
            "lexical": round(lexical, 6),
            "trace_boost": trace,
            "exact_trace_match": exact_frame,
            "owner_trace_match": owner_frame,
            "source_path": candidate.get("source_path"),
        })
    return sorted(results, key=lambda row: (-row["score"], row["index"]))


def shortlist(snapshot: dict, limit: int = 8) -> list[int]:
    """Select a stable, diverse list of method indices, no patch labels."""
    if limit < 2:
        raise ValueError("limit must be at least two")
    ranked = failure_rerank(snapshot)
    selected, seen_methods, per_file = [], set(), {}
    for row in ranked:
        source = row["source_path"] or row["method"].rsplit(".", 1)[0]
        if per_file.get(source, 0) >= 2:
            continue
        if row["method"] in seen_methods:
            continue
        selected.append(row["index"])
        seen_methods.add(row["method"])
        per_file[source] = per_file.get(source, 0) + 1
        if len(selected) >= limit:
            break
    return selected
