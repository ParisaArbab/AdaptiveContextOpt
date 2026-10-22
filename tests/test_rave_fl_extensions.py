"""Tests for live-localizer contract, grounding, study splits and bandit guards.

Synthetic fixtures are used ONLY for unit testing. They are not research data.
"""
from __future__ import annotations

import json
import sys

import pytest

from rave_fl.bandit import train as train_bandit
from rave_fl.bandit import validate_logs
from rave_fl.code_sketch import FEATURE_NAMES, grounding_features, prepare_candidates
from rave_fl.core import Action, Candidate
from rave_fl.localizer_bridge import run_localizer
from rave_fl.pilot_types import CallResult, Observation, PolicyConfig
from rave_fl.rank_repair import RankRepairPilot, RepairConfig
from rave_fl.study import study_split


def _localizer_fixture(tmp_path, candidate_rows):
    script = tmp_path / "localizer.py"
    script.write_text("import json\nprint(json.dumps(" + repr({
        "localizer": "fixture-live-v1", "runtime_pruning": "not_available",
        "candidates": candidate_rows}) + "))\n")
    return {"bug": "Toy-1", "checkout": str(tmp_path),
            "localizer_command": [sys.executable, str(script), "--bug", "{bug}"]}


def test_live_localizer_preserves_measured_semantic_and_coverage(tmp_path):
    case = _localizer_fixture(tmp_path, [
        {"function_id": "p.F.a()", "file_class": "p.F", "start_line": 10,
         "end_line": 20, "semantic_score": .8, "coverage_score": .2,
         "ranker_scores": {"semantic": .8, "coverage": .2}},
        {"function_id": "p.F.b()", "file_class": "p.F", "start_line": 21,
         "end_line": 30, "semantic_score": .4, "coverage_score": .6}])
    candidates, meta, provenance = run_localizer(case)
    assert provenance["mode"] == "live_external"
    assert provenance["runtime_pruning"] == "not_available"
    assert len(candidates) == 2
    assert candidates[0].semantic_score == .8
    assert candidates[0].coverage_score == .2
    assert meta["p.F.a()"]["prior"] == .5
    sources = {c.function_id: f"public int {c.function_id.split('.')[-1].split('(')[0]}() {{ return 3600000; }}"
               for c in candidates}
    prepared = prepare_candidates(candidates, sources, meta, "p.F expected 3600000")
    assert prepared[0].features["semantic_score"] == .8
    assert prepared[0].features["coverage_score"] == .2
    assert prepared[0].features["numeric_evidence_overlap"] == 1


def test_pilot_prepare_case_uses_live_localizer_with_real_source_spans(tmp_path):
    from rave_fl.pilot import prepare_case
    source = tmp_path / "src/main/java/p/F.java"
    source.parent.mkdir(parents=True)
    source.write_text("""class F {
public int a() { return 3600000; }
}
""")
    evidence = tmp_path / "failure.txt"
    evidence.write_text("p.F failed: expected 3600000")
    case = _localizer_fixture(tmp_path, [
        {"function_id": "p.F.a()", "file_class": "p.F", "start_line": 2,
         "end_line": 2, "semantic_score": .8, "coverage_score": .2}])
    case.update({"flexfl_repo": str(tmp_path), "evidence_file": str(evidence)})
    candidates, context, provenance = prepare_case(case)
    assert len(candidates) == 1
    assert "return 3600000" in candidates[0].source
    assert provenance["base_localizer"]["mode"] == "live_external"
    assert provenance["candidate_features"]["p.F.a()"]["coverage_score"] == .2
    assert "expected 3600000" in context


def test_live_localizer_fails_on_invalid_or_duplicated_scores(tmp_path):
    row = {"function_id": "p.F.a()", "file_class": "p.F", "start_line": 10,
           "end_line": 20, "semantic_score": .8, "coverage_score": 1.1}
    with pytest.raises(ValueError, match="normalized score"):
        run_localizer(_localizer_fixture(tmp_path, [row]))
    row["coverage_score"] = .4
    with pytest.raises(ValueError, match="unique"):
        run_localizer(_localizer_fixture(tmp_path, [row, row]))


def test_grounding_numeric_survives_function_name_masking():
    features = grounding_features("p.F.suspectedName()",
                                  "public int suspectedName(){ return 3600000; }",
                                  "p.F Assertion expected 3600000, actual 7200000")
    assert features["numeric_evidence_overlap"] == .5
    assert features["trace_class_match"] == 1


def test_grounded_rescue_full_verifies_even_if_rank_utility_is_zero():
    classes = [Candidate(f"p.F.m{i}()", f"m{i}()", semantic_score=.9-i*.1)
               for i in range(3)]
    sources = {c.function_id: f"public int m{i}() {{ return {i + 100}; }}" for i, c in enumerate(classes)}
    metadata = {c.function_id: {"methods": ["SBIR"], "ranks": {"SBIR": i+1}}
                for i, c in enumerate(classes)}
    candidates = prepare_candidates(classes, sources, metadata, "p.F failed for value 101")

    class FakeBackend:
        def estimate(self, candidates, action, level):
            return 40

        def execute(self, candidates, action, level):
            return CallResult([Observation(c.function_id, .1, .05, "not enough evidence")
                               for c in candidates], 20, 5, 0.01)

    result = RankRepairPilot(candidates, FakeBackend(),
                             PolicyConfig(token_budget=300, batch_size=3, mode="repair_grounded"),
                             RepairConfig(max_full=1)).execute()
    assert result["screened_count"] == 3
    assert result["full_verified_count"] == 1
    assert any(call["action"] == Action.FULL_VERIFY.value for call in result["calls"])


def test_project_heldout_split_is_disjoint_and_repeatable():
    manifest = {"cases": [{"bug": f"{project}-{i}", "project": project,
                           "checkout": f"/checkout/{project}-{i}", "flexfl_repo": "/reference"}
                          for project in ("Time", "Math", "Chart") for i in range(1, 4)]}
    result = study_split(manifest, seed=7)
    again = study_split(manifest, seed=7)
    assert result == again
    partitions = result["project_splits"]
    assert all(partitions[k] for k in ("train", "calibration", "test"))
    assert set(partitions["train"]).isdisjoint(partitions["test"])
    assert sum(len(v) for v in result["splits"].values()) == 9
    manifest["cases"].append(dict(manifest["cases"][0]))
    with pytest.raises(ValueError, match="unique"):
        study_split(manifest)


def _bandit_rows():
    rows = []
    split = {"train": ["A-1", "B-1"], "calibration": ["C-1"], "test": ["D-1"]}
    for project, bug in (("A", "A-1"), ("B", "B-1"), ("C", "C-1"), ("D", "D-1")):
        for rep in range(20):
            for action in ("BATCH_SCREEN", "EXPAND_CONTEXT", "FULL_VERIFY", "STOP"):
                feats = {name: 0.0 for name in FEATURE_NAMES}
                feats["prior"] = .5
                feats["uncertainty"] = .5
                rows.append({"bug": bug, "project": project, "features": feats,
                             "action": action, "propensity": .25,
                             "reward": .5 if action == "FULL_VERIFY" else -.2,
                             "reward_source": "observed_randomized_action",
                             "reward_definition": "test_only_observed_quality_minus_tokens_v1"})
    return rows, split


def test_bandit_offline_training_requires_real_randomized_logged_support():
    rows, split = _bandit_rows()
    artifact = train_bandit(rows, split, min_train_bugs=2, min_test_bugs=1)
    assert not artifact["online_enabled"]
    assert artifact["held_out_test"]["effective_sample_size"] >= 5
    assert set(artifact["models"]) == {"BATCH_SCREEN", "EXPAND_CONTEXT", "FULL_VERIFY", "STOP"}
    rows[0]["propensity"] = 0
    with pytest.raises(ValueError, match="probability"):
        validate_logs(rows, split, min_train_bugs=2, min_test_bugs=1)


def test_bandit_rejects_nonrandomized_and_counterfactual_results():
    rows, split = _bandit_rows()
    rows[0]["reward_source"] = "counterfactual_full_replay"
    with pytest.raises(ValueError, match="real randomized"):
        validate_logs(rows, split, min_train_bugs=2, min_test_bugs=1)
    rows, split = _bandit_rows()
    for row in rows:
        row["propensity"] = 1.0
    with pytest.raises(ValueError, match="Deterministic"):
        validate_logs(rows, split, min_train_bugs=2, min_test_bugs=1)
