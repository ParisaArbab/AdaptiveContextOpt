"""Tests for authentic Defects4J evidence preparation and multi-bug audit.

All tests are local/synthetic and do NOT invoke Defects4J or a model.
"""
import json
from pathlib import Path

import pytest

from scripts.prepare_graft_phase3_real_bugs import (
    _one_test_method, _only_known_cases, _file_labels, inspect,
    find_test_source,
)
from scripts.prepare_graft_phase3_multibug import prepare


def source_manifest(tmp_path):
    path = tmp_path / "frozen.json"
    data = {"cases": [
        {"bug": "Time-1", "buggy_revision": "a" * 40},
        {"bug": "Lang-1", "buggy_revision": "b" * 40},
        {"bug": "Math-1", "buggy_revision": "c" * 40},
    ]}
    path.write_text(json.dumps(data))
    return path


def test_readonly_inspection_never_creates_files_or_calls_provider(tmp_path):
    manifest = source_manifest(tmp_path)
    root = tmp_path / "absent"
    details = inspect(manifest, None, root)
    assert details["new_model_calls"] == 0
    assert not details["network_or_tests_performed"]
    assert details["not_equivalent_to_original_flexfl_condition"]
    assert [b["bug"] for b in details["bugs"]] == ["Time-1", "Lang-1", "Math-1"]
    assert not root.exists()


def test_frozen_cohort_must_not_be_reordered_or_replaced():
    with pytest.raises(ValueError, match="Cohort"):
        _only_known_cases([{"bug": "Time-2"}, {"bug": "Lang-1"}, {"bug": "Math-1"}])


def test_real_java_test_ast_extracts_complete_method_not_partial_excerpt(tmp_path):
    source = (
        "package org.example;\n"
        "public class TestLogic {\n"
        "public void testOne() {\n"
        "  try { target(); fail(); }\n"
        "  catch (IllegalArgumentException expected) { assertTrue(true); }\n"
        "}\n"
        "public void testOther() { assertTrue(true); }\n"
        "}\n"
    ).encode()
    method = _one_test_method(source, "testOne")
    assert method and b"catch (IllegalArgumentException expected)" in method
    assert b"testOther" not in method
    assert _one_test_method(source, "missing") is None

    root = tmp_path / "checkout"
    f = root / "src/test/java/org/example/TestLogic.java"
    f.parent.mkdir(parents=True)
    f.write_bytes(source)
    found, extracted = find_test_source(
        root, "org.example.TestLogic::testOne", "src/test/java",
    )
    assert found == f and extracted == method


def test_modified_class_proxy_labels_require_actual_source_path(tmp_path):
    root = tmp_path / "checkout"
    file = root / "src/main/java/org/example/Logic.java"
    file.parent.mkdir(parents=True)
    file.write_text("package org.example; public class Logic {}")
    assert _file_labels(root, "org.example.Logic\norg.example.Missing", "src/main/java") == [
        "src/main/java/org/example/Logic.java"
    ]


def test_pair_config_requires_three_real_inputs_and_no_flexfl_replacement(tmp_path):
    source = Path("configs/graft_independent_qwen7b_control.json")
    d = tmp_path / "dataset.json"
    cases = []
    for bug in ("Time-1", "Lang-1", "Math-1"):
        evidence = tmp_path / f"{bug}.txt"
        evidence.write_text("VERBATIM original failing test source")
        checkout = tmp_path / bug
        checkout.mkdir()
        cases.append({
            "bug": bug,
            "checkout": str(checkout),
            "evidence_file": str(evidence),
            "evidence_kind": "observed_failure",
            "gold": {"files": ["src/main/java/A.java"], "functions": []},
        })
    d.write_text(json.dumps({
        "cases": cases,
        "not_original_flexfl_condition": True,
    }))
    out = prepare(source, d, tmp_path / "ready")
    cfg = json.loads(out.read_text())
    assert cfg["benchmark"]["candidate_provider"] == "bm25"
    assert cfg["benchmark"]["require_observed_failure"] is True
    assert cfg["benchmark"]["fusions"] == ["rrf", "suspicion_rank_blend"]
    assert cfg["graft_policies"] == [
        "l1_anchor", "l1_graph_focused",
        "l1_graph_test_aware_nostop", "l1_graph_test_aware",
    ]
    assert cfg["benchmark"]["max_new_calls"] == 135
    with pytest.raises(FileExistsError):
        prepare(source, d, tmp_path / "ready")

    data = json.loads(d.read_text())
    data["cases"].pop()
    d.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="three"):
        prepare(source, d, tmp_path / "incomplete")


def test_paired_audit_reports_file_level_only_and_real_stopping_status(tmp_path):
    from scripts.audit_graft_phase3_multibug import inspect

    run = tmp_path / "run"
    run.mkdir()
    policies = (
        "l1_anchor", "l1_graph_focused",
        "l1_graph_test_aware_nostop", "l1_graph_test_aware",
    )
    # Six complete bug/fusion pairs, each with four policies, are required.
    # This fixture exercises audit completeness; it is NOT research evidence.
    sample = []
    for bug in ("Time-1", "Lang-1", "Math-1"):
        for fusion in ("rrf", "suspicion_rank_blend"):
            for policy in policies:
                sample.append({
                    "instance_id": bug,
                    "fusion": fusion,
                    "policy": policy,
                    "status": "completed",
                    "metrics": {"java_strict_v2": {
                        "gold_semantics": "official_d4j_modified_classes_file_proxy",
                        "file": {
                            "first_relevant_rank": 2, "MRR": 0.5,
                            "Top-1": 0, "Top-5": 1,
                            "pool_gold_count": 1,
                            "candidate_pool_coverage": 1.0,
                        },
                        "method": None,
                    }},
                    "arm": {
                        "actual_tokens": 100 if policy == "l1_anchor" else 120,
                        "calls": [{"stage": "sketch", "candidate_ids": ["x.a()"]}]
                        + ([{
                            "stage": "graph", "candidate_ids": ["x.a()"],
                            "selection": None,
                        }] if policy != "l1_anchor" else []),
                        "trace": [{"phase": "route", "options": []}],
                        "stage_histories": {},
                    },
                })
    (run / "results.json").write_text(json.dumps(sample))
    (run / "accounting.json").write_text(json.dumps({
        "new_calls": 1, "known_provider_tokens": 75,
    }))

    summary = inspect(run)
    assert summary["all_pairs_completed"]
    assert len(summary["records"]) == 24
    assert len(summary["paired_comparisons"]) == 6
    assert not summary["any_real_stop_gate_activated"]
    assert summary["gold_is_file_level_proxy_not_method_truth"]
    assert all(row["method_mrr"] is None for row in summary["records"])
    assert all(row["graph_calls_saved_by_stop"] == 0
               for row in summary["paired_comparisons"])
    assert summary["new_model_calls_by_this_audit"] == 0

    # Missing even one policy must be reported, never silently considered complete.
    sample.pop()
    (run / "results.json").write_text(json.dumps(sample))
    incomplete = inspect(run)
    assert not incomplete["all_pairs_completed"]
    assert any(row["status"] == "incomplete_arm_set"
               for row in incomplete["paired_comparisons"])

    # Missing an entire bug/fusion pair must also be incomplete.
    sample = [r for r in sample
              if not (r["instance_id"] == "Math-1"
                      and r["fusion"] == "suspicion_rank_blend")]
    (run / "results.json").write_text(json.dumps(sample))
    incomplete = inspect(run)
    assert not incomplete["all_pairs_completed"]
