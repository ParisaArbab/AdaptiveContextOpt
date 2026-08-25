"""
compression_tax_analyzer.py — WP1

Auto-classifies instances where compression cost localization accuracy,
now across THREE conditions (raw / rtk / lean-ctx) instead of two, so we can
show not just "compression hurts" but "does the smarter compressor hurt
less."

Taxonomy (unchanged from the formal error_taxonomy_report.md):
  T1: Lost value evidence (array/diff dumps)
  T2: Lost stack frames breaking causal chain traversal
  T3: Collapsed sub-failures hiding parametrized test patterns
  T4: Dropped non-error log lines
  T5: Over-summarized failure reasons too generic for discrimination

Any result whose compressor mode == 'reference' (i.e. lean-ctx's real daemon
wasn't reachable and the documented-behavior Python re-implementation ran
instead) is tagged provisional=True and excluded from the headline numbers,
same rule applied to rtk's own reference-mode results in earlier runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class InstanceOutcome:
    instance_id: str
    condition: str  # "raw" | "rtk" | "leanctx"
    compressor_mode: str  # "n/a" for raw, else "daemon" | "reference"
    predicted_files: List[str]
    ground_truth_files: List[str]
    file_level_correct: bool
    taxonomy_tags: List[str] = field(default_factory=list)
    provisional: bool = False


def score_file_level(predicted: List[str], ground_truth: List[str]) -> bool:
    """File-level success: at least one predicted file is in ground truth —
    matches the metric already used for the 0.917 -> 0.833 result."""
    return bool(set(predicted) & set(ground_truth))


def classify_taxonomy(raw_text: str, compressed_text: str) -> List[str]:
    """Lightweight, explainable heuristics for tagging WHY something was
    lost — mirrors the manual classification originally used for
    sympy__sympy-16792 (T1: assertion diff collapsed to identical ellipses)
    and astropy__astropy-12907 (T1/T2: boolean matrix + call-stack path)."""
    tags = []
    raw_lines, comp_lines = raw_text.splitlines(), compressed_text.splitlines()
    comp_set = set(comp_lines)

    if any(("assert" in l or "AssertionError" in l) for l in raw_lines) and \
       compressed_text.count("...") >= 2:
        tags.append("T1_lost_value_evidence")

    raw_frames = sum(1 for l in raw_lines if l.strip().startswith("File \""))
    comp_frames = sum(1 for l in comp_lines if l.strip().startswith("File \""))
    if raw_frames >= 2 and comp_frames <= 1:
        tags.append("T2_lost_stack_frames")

    if raw_text.count("FAILED") >= 3 and compressed_text.count("FAILED") <= 1:
        tags.append("T3_collapsed_subfailures")

    dropped_nonerror = sum(
        1 for l in raw_lines
        if l not in comp_set and "error" not in l.lower() and "assert" not in l.lower() and l.strip()
    )
    if dropped_nonerror > 0.5 * len(raw_lines):
        tags.append("T4_dropped_nonerror_lines")

    if len(compressed_text) < 0.1 * len(raw_text) and "..." in compressed_text:
        tags.append("T5_over_summarized")

    return tags


def analyze(
    results_path: Path,
) -> dict:
    """results_path is a JSON file: list of dicts with keys matching
    InstanceOutcome (produced by run_wp1_benchmark.py)."""
    raw_results = json.loads(results_path.read_text())
    outcomes = [InstanceOutcome(**r) for r in raw_results]

    by_condition: dict[str, list[InstanceOutcome]] = {}
    for o in outcomes:
        by_condition.setdefault(o.condition, []).append(o)

    summary = {}
    for condition, items in by_condition.items():
        trusted = [i for i in items if not i.provisional]
        n = len(trusted)
        correct = sum(1 for i in trusted if i.file_level_correct)
        summary[condition] = {
            "n_instances": n,
            "n_provisional_excluded": len(items) - n,
            "file_level_success_rate": round(correct / n, 3) if n else None,
        }

    # instances where raw succeeded but a compressed condition failed —
    # the actual "compression tax" cases
    tax_cases = {}
    by_instance: dict[str, dict[str, InstanceOutcome]] = {}
    for o in outcomes:
        by_instance.setdefault(o.instance_id, {})[o.condition] = o

    for instance_id, conds in by_instance.items():
        raw_ok = conds.get("raw") and conds["raw"].file_level_correct
        if not raw_ok:
            continue
        for cond_name in ("rtk", "leanctx"):
            cond = conds.get(cond_name)
            if cond and not cond.file_level_correct and not cond.provisional:
                tax_cases.setdefault(cond_name, []).append(
                    {"instance_id": instance_id, "taxonomy_tags": cond.taxonomy_tags}
                )

    return {
        "summary_by_condition": summary,
        "compression_tax_cases": tax_cases,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results/wp1_results.json")
    ap.add_argument("--out", type=str, default="results/compression_tax_report.json")
    args = ap.parse_args()

    report = analyze(Path(args.results))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
