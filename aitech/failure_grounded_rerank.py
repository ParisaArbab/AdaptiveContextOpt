"""Failure-grounded method reranking for frozen buggy-source candidates.

This is a training-free BM25-style lexical score with strong *pre-gold*
stack-frame attribution signals. It is deliberately not called FaR-Loc,
semantic embedding retrieval, causal fault localization, or calibrated ML.

All operations use the failing-test evidence and buggy source excerpts.
"""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[A-Za-z][a-z]*|[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\\b)|[0-9]+")
FRAME = re.compile(r"\\bat\\s+([\\w.$]+)\\.([\\w$<>]+)\\((?:[\\w$]+\\.java):\\d+\\)")
STOP = {"java", "lang", "org", "com", "the", "this", "that", "test",
        "error", "exception", "assert", "public", "void", "return", "class"}


def terms(text: str) -> list[str]:
    # Split code symbols and camel-case words, ignoring punctuation.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\\1 \\2", text)
    return [x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9]*", text)
            if len(x) >= 3 and x.lower() not in STOP]


def rerank(snapshot: dict, *, frame_bonus: float = 9.0,
           method_bonus: float = 4.0) -> list[int]:
    """Rank candidate indices without accessing gold labels.

    Stack frame identity is matched to buggy-source class/method names.
    BM25 is used only as a bounded secondary lexical signal.
    Tie-breaking is original retrieval rank, then index.
    """
    items = snapshot["candidates"]
    if not items:
        return []
    failure = snapshot["context"]
    frames = FRAME.findall(failure)
    frame_classes = {c.replace("$", ".") for c, _ in frames}
    frame_methods = {(c.replace("$", "."), m) for c, m in frames}
    query = Counter(terms(failure))
    documents = [Counter(terms(item["levels"][1][:1400])) for item in items]
    n = len(items)
    df = Counter()
    for doc in documents:
        df.update(doc.keys())
    avg_len = max(1.0, sum(sum(d.values()) for d in documents) / n)

    scored = []
    for i, item in enumerate(items):
        method_id = item["function_id"].split("(", 1)[0]
        owner, method = method_id.rsplit(".", 1)
        if any(owner == cls or owner.endswith("." + cls.split(".")[-1])
               for cls in frame_classes):
            provenance_score = frame_bonus
        else:
            provenance_score = 0.0
        if (owner, method) in frame_methods:
            provenance_score += method_bonus
        doc = documents[i]
        doc_length = sum(doc.values())
        bm25 = 0.0
        for word, q_count in query.items():
            tf = doc.get(word, 0)
            if not tf:
                continue
            idf = max(0.0, math.log(1.0 + (n - df[word] + 0.5) / (df[word] + 0.5)))
            bm25 += idf * (tf * 2.2 / (tf + 1.2 * (
                0.25 + 0.75 * doc_length / avg_len))) * min(q_count, 2)
        # Avoid allowing repeated generic stack words to dominate identity.
        scored.append((-(provenance_score + 0.45 * bm25), i))
    return [i for _, i in sorted(scored)]


def shortlist(snapshot: dict, order: list[int], count: int = 8) -> dict:
    if not 1 <= count <= len(order):
        raise ValueError("invalid shortlist size")
    if len(order) != len(set(order)) or any(
            i < 0 or i >= len(snapshot["candidates"]) for i in order):
        raise ValueError("invalid original candidate order")
    result = dict(snapshot)
    result["candidates"] = [snapshot["candidates"][i] for i in order[:count]]
    result["retrieval_original_indices"] = order[:count]
    return result
