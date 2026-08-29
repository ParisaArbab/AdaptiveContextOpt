"""Method-level Top-k, AP and reciprocal-rank evaluation for FlexFL outputs."""
from __future__ import annotations

import re


def _simple_type(value: str) -> str:
    value = re.sub(r"<.*?>", "", value).strip().replace("...", "[]")
    value = value.rsplit(".", 1)[-1]
    return value


def parse_method(value: str) -> tuple[str, str, tuple[str, ...]]:
    text = re.sub(r"\s+", "", value or "").replace("#", ".").replace("$", ".")
    if "(" in text:
        head, tail = text.split("(", 1)
        params_raw = tail.rsplit(")", 1)[0]
        params = tuple(_simple_type(p) for p in params_raw.split(",") if p)
    else:
        head, params = text, ()
    if "." in head:
        cls, method = head.rsplit(".", 1)
    else:
        cls, method = "", head
    return cls.lower(), method.lower(), tuple(p.lower() for p in params)


def method_matches(prediction: str, truth: str) -> bool:
    pc, pm, pp = parse_method(prediction)
    tc, tm, tp = parse_method(truth)
    if pm != tm:
        return False
    if pc and tc and pc != tc:
        if pc.replace(".", "") != tc.replace(".", ""):
            return False
    if pp and tp and len(pp) == len(tp):
        return all(a == b for a, b in zip(pp, tp))
    return not (pp and tp and len(pp) != len(tp))


def first_relevant_rank(predictions: list[str], truths: list[str]) -> int | None:
    for rank, pred in enumerate(predictions, 1):
        if any(method_matches(pred, truth) for truth in truths):
            return rank
    return None


def evaluate_top5(predictions: list[str], truths: list[str]) -> dict:
    preds = predictions[:5]
    rank = first_relevant_rank(preds, truths)
    relevant_seen = 0
    precision_sum = 0.0
    matched_truths: set[int] = set()
    for i, pred in enumerate(preds, 1):
        newly_matched = None
        for j, truth in enumerate(truths):
            if j not in matched_truths and method_matches(pred, truth):
                newly_matched = j
                break
        if newly_matched is not None:
            matched_truths.add(newly_matched)
            relevant_seen += 1
            precision_sum += relevant_seen / i
    denom = max(1, len(truths))
    ap = precision_sum / denom
    return {
        "top1": bool(rank and rank <= 1),
        "top3": bool(rank and rank <= 3),
        "top5": bool(rank and rank <= 5),
        "first_relevant_rank": rank,
        "average_precision": ap,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "MAP": 0.0, "MRR": 0.0}
    n = len(rows)
    return {
        "n": n,
        "top1": sum(bool(r["top1"]) for r in rows) / n,
        "top3": sum(bool(r["top3"]) for r in rows) / n,
        "top5": sum(bool(r["top5"]) for r in rows) / n,
        "MAP": sum(float(r["average_precision"]) for r in rows) / n,
        "MRR": sum(float(r["reciprocal_rank"]) for r in rows) / n,
    }
