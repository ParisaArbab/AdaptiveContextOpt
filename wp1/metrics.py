"""
metrics.py — WP1 step 2b: rank-aware localization scoring.

Replaces the old binary `score_file_level` ("did any predicted file appear in
ground truth"), which had two problems: it ignored rank entirely, and it
rewarded breadth — GraphLocator expansion adds entities to the prediction
set, so a wider prediction could only ever score better, never worse.

This implements FlexFL's own metric set (eval_FL.py: Top-1/Top-3/Top-5 hit
rate plus MAP and MRR), computed at BOTH granularities:

  method level — the paper's granularity, and the one that actually
    distinguishes a good localization from a lucky file hit.
  file level   — kept because SWE-bench ground truth is most reliable at
    file granularity (gold-patch symbol extraction is regex-based and can
    miss a symbol; the file list cannot be missed).

Precision is reported alongside, so expansion that adds noise is visible
instead of free.

Symbol matching is deliberately forgiving about the *shape* of a name but
strict about the file: Graphify labels an entity `Class.method` or `method`
depending on the grammar, while gold-patch parsing yields the bare `def`/
`class` name. Requiring the file to match and the name to match on its last
dotted segment is the tightest rule that doesn't systematically under-count
real hits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

TOP_KS = (1, 3, 5)


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./").lower()


def split_symbol(symbol: str) -> tuple[str, str]:
    """'src/foo.py::Bar.baz' -> ('src/foo.py', 'bar.baz')

    Graphify labels callables with a trailing call suffix — `normalize()`,
    and for Java a full signature like `add(int,int)` — while gold-patch
    parsing yields the bare name. Without stripping it, NOTHING ever
    matches and every method-level score is zero regardless of how good the
    localization was. Caught by the smoke test, where the correct method
    was ranked 3rd and still scored 0."""
    if "::" in symbol:
        path, name = symbol.split("::", 1)
    else:
        path, name = "", symbol
    name = name.strip()
    if "(" in name:
        name = name[: name.index("(")]
    return _norm_path(path), name.strip().lower()


def symbol_matches(predicted: str, truth: str) -> bool:
    p_file, p_name = split_symbol(predicted)
    t_file, t_name = split_symbol(truth)
    if p_file and t_file and p_file != t_file:
        return False
    if p_name == t_name:
        return True
    # Class.method vs method, or module.Class vs Class
    return p_name.split(".")[-1] == t_name.split(".")[-1]


def _first_hit_rank(predicted: Sequence[str], truth: Sequence[str]) -> Optional[int]:
    for rank, p in enumerate(predicted, 1):
        if any(symbol_matches(p, t) for t in truth):
            return rank
    return None


def reciprocal_rank(predicted: Sequence[str], truth: Sequence[str]) -> float:
    rank = _first_hit_rank(predicted, truth)
    return 1.0 / rank if rank else 0.0


def average_precision(predicted: Sequence[str], truth: Sequence[str]) -> float:
    """AP over the ranked prediction list. Denominator is |truth|, so failing
    to retrieve a buggy method at all is penalised — matching eval_FL.py
    rather than the (more generous) retrieved-only variant."""
    if not truth:
        return 0.0
    hits = 0
    precision_sum = 0.0
    matched_truths: List[str] = []
    for rank, p in enumerate(predicted, 1):
        for t in truth:
            if t in matched_truths:
                continue
            if symbol_matches(p, t):
                matched_truths.append(t)
                hits += 1
                precision_sum += hits / rank
                break
    return precision_sum / len(truth)


def score_ranked(predicted: Sequence[str], truth: Sequence[str]) -> Dict[str, float]:
    """Per-instance scores for one ranked prediction list."""
    predicted = list(predicted)
    truth = list(truth)
    rank = _first_hit_rank(predicted, truth)
    matched = sum(1 for p in predicted if any(symbol_matches(p, t) for t in truth))
    out: Dict[str, float] = {}
    for k in TOP_KS:
        out[f"top{k}"] = 1.0 if (rank is not None and rank <= k) else 0.0
    out["mrr"] = 1.0 / rank if rank else 0.0
    out["ap"] = average_precision(predicted, truth)
    out["precision"] = matched / len(predicted) if predicted else 0.0
    out["recall"] = (
        len([t for t in truth if any(symbol_matches(p, t) for p in predicted)]) / len(truth)
        if truth else 0.0
    )
    out["n_predicted"] = float(len(predicted))
    return out


def files_from_symbols(symbols: Iterable[str]) -> List[str]:
    """Rank-preserving, de-duplicated file list from a ranked symbol list —
    the first time a file appears is its rank, which is what file-level
    Top-k should score against."""
    seen: List[str] = []
    for s in symbols:
        path = s.split("::", 1)[0] if "::" in s else s
        if path and path not in seen:
            seen.append(path)
    return seen


def score_instance(
    predicted_functions: Sequence[str],
    predicted_files: Sequence[str],
    truth_functions: Sequence[str],
    truth_files: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Both granularities in one call. `truth_functions` may be empty (patch
    parsing found no symbol) — in that case the method-level block is
    returned as None so it can be excluded from aggregation rather than
    silently scoring 0 and dragging the mean down."""
    return {
        "method_level": score_ranked(predicted_functions, truth_functions) if truth_functions else None,
        "file_level": score_ranked(predicted_files, truth_files) if truth_files else None,
    }


def aggregate(per_instance: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean over instances. Top-k means are hit rates, mean of AP is MAP,
    mean of RR is MRR — the standard reading of eval_FL.py's outputs."""
    usable = [p for p in per_instance if p]
    if not usable:
        return {}
    keys = usable[0].keys()
    agg = {k: round(sum(p[k] for p in usable) / len(usable), 4) for k in keys}
    agg["map"] = agg.pop("ap", 0.0)
    agg["n"] = len(usable)
    return agg
