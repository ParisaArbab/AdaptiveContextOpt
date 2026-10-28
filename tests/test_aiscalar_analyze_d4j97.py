"""Deterministic offline checks for 97-bug benchmark aggregation.

Fixtures are synthetic unit-test inputs only; they are never benchmark results.
"""
import json
from pathlib import Path

from scripts.aiscalar_analyze_d4j97 import load_cases, compute, write_details


def row(bug, policy, status, tokens=None, top1=None, top5=None, mrr=None):
    method = ({"Top-1": top1, "Top-5": top5, "MRR": mrr}
              if top1 is not None else None)
    return {
        "instance_id": bug,
        "policy": policy,
        "status": status,
        "accounted_tokens": tokens,
        "metrics": {"method": method} if method else {},
        "synthetic": False,
    }


def test_matched_and_end_to_end_metrics_do_not_mix_failure_tokens(tmp_path):
    a = [
        row("Bug-1", "full", "completed", 100, 1, 1, 1),
        row("Bug-1", "fixed_l1", "completed", 70, 0, 1, .5),
        row("Bug-1", "v2_utility", "completed", 50, 0, 1, .5),
    ]
    b = [
        row("Bug-2", "full", "failed", 10),
        row("Bug-2", "fixed_l1", "completed", 20, 0, 0, 0),
        row("Bug-2", "v2_utility", "completed", 30, 1, 1, 1),
    ]
    c = [
        row("Bug-3", "full", "incomplete", 2),
        row("Bug-3", "fixed_l1", "incomplete", 0),
        row("Bug-3", "v2_utility", "incomplete", 0),
    ]
    root = tmp_path / "source"
    for bug, rows in [("Bug-1", a), ("Bug-2", b), ("Bug-3", c)]:
        p = root / bug
        p.mkdir(parents=True)
        (p / "results.json").write_text(json.dumps(rows))
    cases, missing = load_cases(root)
    assert not missing
    result, complete, paired_cost, paired_accuracy = compute(cases, "method")
    assert result["n_bug_directories"] == 3
    assert result["n_all_three_completed"] == 1
    assert result["n_paired_cost"] == 1
    assert len(complete) == len(paired_cost) == len(paired_accuracy) == 1
    assert result["n_bugs_with_at_least_one_label"] == 2
    assert result["policies"]["v2_utility"]["paired_total_accounted_tokens"] == 50
    assert result["policies"]["v2_utility"]["paired_cost_reduction_fraction_vs_full"] == .5
    assert result["policies"]["full"]["end_to_end_metrics"]["Top-1"] == .5
    assert result["policies"]["v2_utility"]["end_to_end_metrics"]["Top-5"] == 1
    assert result["policies"]["full"]["statuses"]["failed"] == 1
    out = tmp_path / "summary"
    out.mkdir()
    assert write_details(cases, out, "method").exists()


def test_duplicate_policy_is_rejected(tmp_path):
    p = tmp_path / "source" / "Bug-1"
    p.mkdir(parents=True)
    r = row("Bug-1", "full", "completed", 100, 1, 1, 1)
    (p / "results.json").write_text(json.dumps([r, r]))
    import pytest
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(tmp_path / "source")


def test_synthetic_benchmark_row_is_rejected(tmp_path):
    p = tmp_path / "source" / "Bug-1"
    p.mkdir(parents=True)
    r = row("Bug-1", "full", "completed", 100, 1, 1, 1)
    r["synthetic"] = True
    (p / "results.json").write_text(json.dumps([r]))
    import pytest
    with pytest.raises(ValueError, match="Synthetic"):
        load_cases(tmp_path / "source")
