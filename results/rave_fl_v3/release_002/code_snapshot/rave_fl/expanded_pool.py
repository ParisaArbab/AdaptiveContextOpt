"""Read larger, real SBIR/Ochiai/BoostN candidate pools from FlexFL CSV files.

Do not use labels or gold results for candidate selection. CSV start/end lines
are preserved for extracting the corresponding buggy-checkout Java methods.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from rave_fl.core import Candidate
from wp1.flexfl_data import TRADITIONAL_METHODS, flexfl_root, read_ground_truth
from wp1.evaluation import method_matches


@dataclass(frozen=True)
class RankedMethod:
    function_id: str
    file_class: str
    start_line: int
    end_line: int
    method: str
    local_rank: int
    prior: float


def read_ranked(repo: Path, method: str, bug: str, limit: int = 10) -> list[RankedMethod]:
    if method not in TRADITIONAL_METHODS:
        raise ValueError(f"Unsupported ranking method: {method}")
    if limit < 1:
        raise ValueError("limit must be positive")
    root = flexfl_root(repo)
    path = root / "data" / "FL_results" / method / "Defects4J" / f"{bug}_method-susps.csv"
    if not path.is_file():
        raise FileNotFoundError(f"FlexFL result missing: {path}")
    rows = []
    seen = set()
    with path.open(newline="", errors="replace") as handle:
        for row in csv.DictReader(handle):
            cls = (row.get("File") or "").strip()
            sig = (row.get("Signature") or "").strip()
            if not cls or not sig:
                continue
            name = f"{cls}.{sig}"
            if name in seen:
                continue
            seen.add(name)
            start, end = int(row["StartLine"]), int(row["EndLine"])
            if not (0 < start <= end):
                raise ValueError(f"Invalid source line range for {name}")
            rows.append(RankedMethod(name, cls, start, end, method,
                                     len(rows) + 1, 1 - len(rows) / limit))
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No usable candidates in {path}")
    return rows


def union_pool(repo: Path, bug: str, top_per_method: int = 10) -> tuple[list[Candidate], dict]:
    """Combine three localizers on rank-proxy scores; first occurrence keeps span.

    The rank proxy is NOT semantic similarity, coverage, or a learned risk.
    """
    merged: dict[str, RankedMethod] = {}
    methods: dict[str, list[str]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for method in TRADITIONAL_METHODS:
        for item in read_ranked(repo, method, bug, top_per_method):
            methods.setdefault(item.function_id, []).append(method)
            ranks.setdefault(item.function_id, {})[method] = item.local_rank
            old = merged.get(item.function_id)
            if old is None or item.prior > old.prior:
                merged[item.function_id] = item

    ordered = sorted(merged.values(), key=lambda x: (
        -x.prior, -len(methods[x.function_id]), x.local_rank, x.function_id))
    candidates = [
        Candidate(item.function_id, item.function_id,
                  semantic_score=item.prior, coverage_score=0)
        for item in ordered
    ]
    metadata = {
        item.function_id: {
            "file_class": item.file_class, "start_line": item.start_line,
            "end_line": item.end_line, "methods": methods[item.function_id],
            "rank_proxy": item.prior,
            "ranks": ranks[item.function_id], "top_per_method": top_per_method,
        }
        for item in ordered
    }
    return candidates, metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flexfl-repo", type=Path, required=True)
    p.add_argument("--bug", required=True)
    p.add_argument("--top-per-method", type=int, default=10)
    p.add_argument("--diagnose", action="store_true",
                   help="Offline diagnostic only: read labels and show gold rank")
    args = p.parse_args()
    candidates, metadata = union_pool(args.flexfl_repo, args.bug, args.top_per_method)
    report: dict = {
        "bug": args.bug, "top_per_method": args.top_per_method,
        "unique_candidates": len(candidates),
        "ranked_candidates": [c.function_id for c in candidates],
    }
    if args.diagnose:
        truth = read_ground_truth(args.flexfl_repo, args.bug)
        report["ground_truth"] = truth
        report["ground_truth_rank"] = next(
            (i for i, c in enumerate(candidates, 1)
             if any(method_matches(c.function_id, t) for t in truth)), None)
        report["note"] = "Gold used only here for offline candidate-recall diagnosis."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
