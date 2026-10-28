import json
from pathlib import Path

from scripts.aiscalar_audit_v2_top5_changes import analyze, audit_case


def make_arm(initial_order, final_order, gold, inspected):
    states = {
        ident: {"level": 2 if ident in inspected else 1,
                "observation": {"suspicion": 0.4, "uncertainty": 0.2}}
        for ident in final_order
    }
    return {
        "initial_l1": {"ranked_functions": initial_order},
        "ranked_functions": final_order,
        "states": states,
        "stop_reason": "full_repair_limit",
        "trace": [{"action": "FULL_VERIFY", "ids": inspected}],
        "repair_diagnostics": [
            {"options": [{"function_id": gold, "utility_per_token": 0.002}]}
        ],
    }


def test_loss_audit_uses_offline_gold_to_check_l1_and_full_allocation():
    a = "pkg.Foo.good(int)"
    b = "pkg.Foo.bad(int)"
    x = "pkg.Foo.other(int)"
    c = "pkg.Foo.third(int)"
    d = "pkg.Foo.fourth(int)"
    e = "pkg.Foo.fifth(int)"
    final = [b, x, c, d, e, a]
    initial = [a, b, x, c, d, e]
    full = {
        "policy": "full", "status": "completed",
        "accounted_tokens": 1000, "full_calls": 6,
        "metrics": {"method": {"Top-5": 1, "first_relevant_rank": 1}},
        "arm": {"ranked_functions": initial},
    }
    v2 = {
        "policy": "v2_utility", "status": "completed",
        "accounted_tokens": 750, "full_calls": 1,
        "metrics": {"method": {"Top-5": 0, "first_relevant_rank": 6}},
        "arm": make_arm(initial, final, a, [b]),
    }
    data = audit_case("Foo-1", {"full": full, "v2_utility": v2}, [a])
    assert data["outcome"] == "loss"
    assert data["v2_initial_l1_fault_rank"] == 1
    assert data["v2_fault_rank_from_ids"] == 6
    assert data["full_fault_rank_from_ids"] == 1
    assert data["v2_full_gold_inspections"] == 0
    assert data["v2_token_delta_vs_full"] == -250
    assert data["v2_stop_reason"] == "full_repair_limit"


def test_analyze_skips_cases_without_gold_and_never_writes(tmp_path):
    root = tmp_path / "run"
    bug = root / "Foo-1"
    bug.mkdir(parents=True)
    (bug / "results.json").write_text(json.dumps([
        {"policy": "full", "status": "completed",
         "metrics": {"method": {"Top-5": 1}}},
        {"policy": "v2_utility", "status": "completed",
         "metrics": {"method": {"Top-5": 0}}},
    ]))
    rows, skipped = analyze(root)
    assert rows == []
    assert skipped == ["Foo-1"]
    assert not list(root.rglob("*.csv"))


def test_direct_script_invocation_loads_repository_packages():
    # Regression: executing as a standalone script previously could not import wp1.
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / 'scripts' / 'aiscalar_audit_v2_top5_changes.py'), '--help'],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '--root' in result.stdout
