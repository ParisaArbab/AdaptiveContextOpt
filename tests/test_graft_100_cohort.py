"""Unit tests for a 100-bug predeclared real-benchmark workflow.

All tests use fake Defects4J metadata or synthetic filesystem fixtures.
They invoke no Defects4J checkout, Java tests, Ollama, or network service.
"""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import graft_100_dataset as cohort
from scripts import graft_100_run as batch_runner


def make_cohort(tmp_path, monkeypatch):
    binary = tmp_path / "defects4j" / "framework" / "bin" / "defects4j"
    binary.parent.mkdir(parents=True)
    binary.write_text("synthetic binary fixture\n")
    monkeypatch.setattr(cohort.shutil, "which", lambda cmd: str(binary))
    monkeypatch.setattr(
        cohort.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40, returncode=0)
    )
    monkeypatch.setattr(
        cohort, "official_active_ids", lambda cmd, project: list(range(1, 101))
    )
    root = tmp_path / "study"
    manifest = cohort.freeze("defects4j", root, seed=42)
    return root, manifest


def test_freeze_100_exact_unique_active_bugs_before_experiment(tmp_path, monkeypatch):
    root, frozen = make_cohort(tmp_path, monkeypatch)
    cohort.validate_cohort(frozen)
    assert frozen["total_predeclared_bugs"] == 100
    assert Counter(r["project"] for r in frozen["bugs"]) == dict.fromkeys(cohort.PROJECTS, 20)
    assert len({r["bug"] for r in frozen["bugs"]}) == 100
    assert all(r["bug"] not in cohort.PREVIOUS_DEVELOPMENT for r in frozen["bugs"])
    assert len([r for r in frozen["bugs"] if r["split"] == "evaluation"]) == 80
    assert len([r for r in frozen["bugs"] if r["split"] == "development"]) == 20
    assert [r["project"] for r in cohort.expected_batch(frozen, 3)] == list(cohort.PROJECTS)
    assert (root / "cohort.json").is_file()
    with pytest.raises(FileExistsError):
        cohort.freeze("defects4j", root, seed=42)


def test_freeze_deterministic_and_tamper_evident(tmp_path, monkeypatch):
    _, frozen = make_cohort(tmp_path, monkeypatch)
    changed = json.loads(json.dumps(frozen))
    changed["bugs"][0]["bug"] = changed["bugs"][0]["project"] + "-999"
    with pytest.raises(ValueError, match="selection"):
        cohort.validate_cohort(changed)
    with pytest.raises(ValueError, match="Batch"):
        cohort.expected_batch(frozen, 20)
    with pytest.raises(ValueError):
        cohort.validate_cohort({"schema": cohort.SCHEMA, "bugs": []})


def test_failed_genuine_test_is_not_replaced_with_another_bug(tmp_path, monkeypatch):
    root, frozen = make_cohort(tmp_path, monkeypatch)
    selected = cohort.expected_batch(frozen, 0)
    failed = selected[2]["bug"]
    called = []

    def fake_collect(row, cmd, prep, max_bytes, test_scope):
        assert test_scope == "official_trigger"
        bug = row["bug"]
        called.append(bug)
        if bug == failed:
            raise RuntimeError("fixture: official test failed to prepare")
        checkout = prep / "checkouts" / bug
        checkout.mkdir(parents=True)
        failure = prep / "evidence" / (bug + ".txt")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text("Synthetic test fixture")
        from hashlib import sha256
        case = {
            "bug": bug,
            "checkout": str(checkout),
            "evidence_file": str(failure),
            "evidence_kind": "observed_failure",
            "gold": {"files": ["src/main/java/Example.java"], "functions": []},
        }
        audit = {
            "bug": bug, "test_method": "synthetic::testExample",
            "evidence_sha256": sha256(failure.read_bytes()).hexdigest(),
        }
        return case, audit

    monkeypatch.setattr(cohort, "collect_case", fake_collect)
    output = cohort.collect_batch(root, frozen, 0, "fake-defects4j")
    assert called == [r["bug"] for r in selected]
    assert output["status"] == "incomplete"
    assert output["predeclared_bug_ids"] == called
    assert [entry["bug"] for entry in output["errors"]] == [failed]
    assert not (root / "batches" / "batch_00" / "prepared" / "manifest.json").exists()
    assert len(output["prepared_bug_ids"]) == 4
    # Already prepared records are not rechecked, and the failed ID stays
    # unavailable on the next attempt. No outcome-driven replacement.
    output2 = cohort.collect_batch(root, frozen, 0, "fake-defects4j")
    assert output2["status"] == "incomplete"
    assert len(output2["prepared_bug_ids"]) == 4
    assert called.count(failed) == 2
    for other in called[:5]:
        if other != failed:
            assert called.count(other) == 1


def test_all_five_prepared_cases_keep_original_cohort_order(tmp_path, monkeypatch):
    root, frozen = make_cohort(tmp_path, monkeypatch)

    def fake_collect(row, cmd, prep, max_bytes, test_scope):
        assert test_scope == "official_trigger"
        bug = row["bug"]
        checkout = prep / "checkouts" / bug
        checkout.mkdir(parents=True)
        evidence = prep / "evidence" / (bug + ".txt")
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("Synthetic test fixture")
        from hashlib import sha256
        case = {
            "bug": bug,
            "checkout": str(checkout),
            "evidence_file": str(evidence),
            "evidence_kind": "observed_failure",
            "gold": {"files": ["src/main/java/Example.java"], "functions": []},
        }
        return case, {
            "bug": bug, "test_method": "synthetic::method",
            "evidence_sha256": sha256(evidence.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(cohort, "collect_case", fake_collect)
    report = cohort.collect_batch(root, frozen, 0, "fake-defects4j")
    assert report["status"] == "ready"
    prepared = root / "batches" / "batch_00" / "prepared" / "manifest.json"
    data = json.loads(prepared.read_text())
    assert data["cohort_selection_sha256"] == frozen["selection_sha256"]
    assert [r["bug"] for r in data["cases"]] == [r["bug"] for r in cohort.expected_batch(frozen, 0)]
    assert all(r["gold"]["functions"] == [] for r in data["cases"])


def test_five_bug_gate_requires_actual_graph_calls_and_real_usage():
    expected = ["Lang-2", "Math-2", "Time-2", "Chart-2", "Closure-2"]
    records = [
        {
            "bug": b, "fusion": f, "policy": p,
            "status": "completed",
            "graph_call_count": 0 if p == "l1_anchor" else 2,
            "file_mrr": 1.0,
        }
        for b in expected for f in batch_runner.FUSIONS
        for p in batch_runner.POLICIES
    ]
    pairs = [
        {"bug": b, "fusion": f, "status": "paired"}
        for b in expected for f in batch_runner.FUSIONS
    ]
    fixture = {
        "records": records,
        "paired_comparisons": pairs,
        "all_pairs_completed": True,
        "physical_provider_accounting": {
            "synthetic": False, "pending": False, "usage_unknown": False,
            "simulated_provider_tokens": 0, "new_calls": 25,
            "physical_inference_tokens": 65000,
        }
    }
    assert batch_runner.check_batch(fixture, expected)["completed"] is True
    records[1]["status"] = "incomplete"
    assert batch_runner.check_batch(fixture, expected)["completed"] is False
    records[1]["status"] = "completed"
    focus = next(r for r in records if r["policy"] == "l1_graph_focused")
    focus["graph_call_count"] = 0
    assert batch_runner.check_batch(fixture, expected)["completed"] is False
    focus["graph_call_count"] = 2
    fixture["physical_provider_accounting"]["usage_unknown"] = True
    assert batch_runner.check_batch(fixture, expected)["completed"] is False


def test_aggregate_never_reports_missing_batches_as_success(tmp_path, monkeypatch):
    root, frozen = make_cohort(tmp_path, monkeypatch)
    result = batch_runner.aggregate(root, frozen)
    assert result["n_predeclared_unique_bugs"] == 100
    assert result["physical_new_usage_across_batches"]["new_calls"] == 0
    for row in result["summary"]:
        assert row["fully_paired_graph_n"] == 0
        assert row["mean_file_mrr_paired_only"]["l1_anchor"] is None
        assert row["predeclared_n"] in (20, 80)
    assert len(result["instances"]) == 200  # 100 frozen bugs, two fusions
    assert (root / "aggregate" / "aggregate.md").is_file()


def test_official_active_ids_rejects_non_integer_output(monkeypatch):
    monkeypatch.setattr(
        cohort.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="Bug IDs: 1 2 3", stderr="", returncode=0,
        )
    )
    with pytest.raises(ValueError, match="Unexpected official active-bug output"):
        cohort.official_active_ids("defects4j", "Chart")
