"""Offline checks that prevent an incomplete Phase-3 run being called done."""
from scripts.verify_graft_phase3_complete import check


BUGS = ("Lang-1", "Math-1", "Time-1")
FUSIONS = ("rrf", "suspicion_rank_blend")
POLICIES = (
    "l1_anchor",
    "l1_graph_focused",
    "l1_graph_test_aware_nostop",
    "l1_graph_test_aware",
)


def audit_fixture():
    records = []
    pairs = []
    for bug in BUGS:
        for fusion in FUSIONS:
            pairs.append({"bug": bug, "fusion": fusion, "status": "paired"})
            for policy in POLICIES:
                records.append({
                    "bug": bug,
                    "fusion": fusion,
                    "policy": policy,
                    "status": "completed",
                    "graph_call_count": 1 if policy != "l1_anchor" else 0,
                })
    return {
        "schema": "graft-phase3-multibug-audit-v1",
        "records": records,
        "paired_comparisons": pairs,
        "all_pairs_completed": True,
        "physical_provider_accounting": {
            "synthetic": False,
            "usage_unknown": False,
            "pending": False,
            "new_calls": 25,
            "known_provider_tokens": 55000,
        },
    }


def test_gate_succeeds_only_for_real_complete_graph_comparison():
    got = check(audit_fixture())
    assert got["completed_real_graph_comparison"] is True
    assert got["new_model_calls_for_gate"] == 0


def test_gate_fails_when_any_policy_is_incomplete():
    a = audit_fixture()
    a["records"][0]["status"] = "incomplete"
    got = check(a)
    assert got["completed_real_graph_comparison"] is False
    assert len(got["incomplete_policy_rows"]) == 1


def test_gate_requires_real_graph_calls_in_focused_and_nostop():
    a = audit_fixture()
    row = next(r for r in a["records"] if r["policy"] == "l1_graph_focused")
    row["graph_call_count"] = 0
    got = check(a)
    assert got["completed_real_graph_comparison"] is False
    assert len(got["graph_focused_or_nostop_without_graph_calls"]) == 1


def test_gate_accepts_zero_test_aware_calls_if_stop_saved_them():
    a = audit_fixture()
    for row in a["records"]:
        if row["policy"] == "l1_graph_test_aware":
            row["graph_call_count"] = 0
    assert check(a)["completed_real_graph_comparison"] is True


def test_gate_rejects_missing_arm_and_unknown_provider_usage():
    a = audit_fixture()
    a["records"].pop()
    assert check(a)["completed_real_graph_comparison"] is False
    b = audit_fixture()
    b["physical_provider_accounting"]["usage_unknown"] = True
    assert check(b)["completed_real_graph_comparison"] is False


def test_gate_does_not_trust_audit_flag_without_all_six_pairs():
    a = audit_fixture()
    a["paired_comparisons"].pop()
    assert check(a)["completed_real_graph_comparison"] is False
