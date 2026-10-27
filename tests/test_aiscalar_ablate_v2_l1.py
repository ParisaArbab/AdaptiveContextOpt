"""Synthetic unit tests, not benchmark results, for the offline 8K ablation."""
import json
from pathlib import Path

from scripts.aiscalar_ablate_v2_l1 import collect, summarize, save


def fixture_case(root, bug, initial, final, label, initial_cost=60, total_cost=100):
    dir_ = root / bug
    dir_.mkdir(parents=True)
    gold = dir_ / "instances" / f"{bug}_test"
    gold.mkdir(parents=True)
    (gold / "evaluation_only.json").write_text(
        json.dumps({"functions": [label], "files": []})
    )
    (dir_ / "results.json").write_text(json.dumps([
        {
            "policy": "v2_utility",
            "status": "completed",
            "accounted_tokens": total_cost,
            "full_calls": 1,
            "metrics": {"method": {"Top-5": 1, "first_relevant_rank": 1}},
            "arm": {
                "ranked_functions": final,
                "initial_l1": {"ranked_functions": initial, "tokens_used": initial_cost},
                "stop_reason": "full_repair_limit",
            },
        }
    ]))


def test_l1_to_v2_repair_gain_and_paid_increment(tmp_path):
    root = tmp_path / "input"
    g = "pkg.C.target(int)"
    candidates = [f"pkg.C.dummy{i}(int)" for i in range(5)] + [g]
    fixture_case(root, "A-1", candidates, [g] + candidates[:5], g)
    rows, excluded = collect(root)
    assert excluded == {}
    assert len(rows) == 1
    r = rows[0]
    assert r["l1_rank"] == 6
    assert r["v2_rank"] == 1
    assert r["l1_to_v2"] == "gain"
    assert r["v2_post_l1_tokens"] == 40
    summary = summarize(rows, excluded)
    assert summary["top5_transitions"]["gain"] == 1
    assert summary["v2"]["total_incremental_tokens_after_l1"] == 40
    out = tmp_path / "derived"
    save(rows, summary, out, False)
    assert (out / "summary.json").exists()
    assert (out / "per_bug_l1_v2.csv").exists()


def test_excludes_incomplete_and_missing_offline_gold(tmp_path):
    root = tmp_path / "input"
    bad = root / "A-1"
    bad.mkdir(parents=True)
    (bad / "results.json").write_text(json.dumps([
        {"policy": "v2_utility", "status": "incomplete"}
    ]))
    empty = root / "B-1"
    empty.mkdir()
    (empty / "results.json").write_text(json.dumps([
        {"policy": "v2_utility", "status": "completed",
         "arm": {"initial_l1": {"ranked_functions": ["pkg.C.x()"]},
                 "ranked_functions": ["pkg.C.x()"]}}
    ]))
    rows, excluded = collect(root)
    assert rows == []
    assert excluded["v2_not_completed"] == 1
    assert excluded["missing_gold"] == 1
