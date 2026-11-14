"""Zero-provider-call checks for target-focused, provenance-preserving retrieval."""
import json
from pathlib import Path

from rave_fl.graft.focused import FOCUSED_SELECTOR, call_names, focus_select
from rave_fl.graft.graph import Graph
from rave_fl.graft.selection import ByteCounter, package, relevance_select
from test_graft_graph import fixture_graph


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "results/graft_phase1_live_20261113_174425/run/instances/Time-4_0d6a675374"


def test_tiny_budget_and_unresolved_call_fail_closed():
    _, _, graph = fixture_graph()
    target = "x.A.a()"
    missing = focus_select(graph, target, 4096, missing_evidence="unknownMethod() call")
    assert missing.status == "infeasible"
    assert missing.reason == "focused_evidence_incomplete"
    assert missing.diagnostics["unresolved_callees"][0]["name"] == "unknownMethod"
    assert not missing.diagnostics["gold_used"]
    low = focus_select(graph, target, 1, missing_evidence="")
    assert low.status == "infeasible"
    assert low.reason == "focused_mandatory_exceeds_budget"


def test_focus_uses_only_real_call_syntax():
    assert call_names("null check in indexOf() method") == ["indexOf"]
    assert call_names("some code is missing") == []
    assert call_names("if (x) in method; realCall()") == ["realCall"]


def test_time4_saved_l1_drives_complete_target_and_callee_package():
    graph = Graph.from_dict(json.loads((SNAPSHOT / "graph.json").read_text()))
    record = next(SNAPSHOT.glob("l1_anchor__rrf_*.json"))
    arm = json.loads(record.read_text())["arm"]

    # Choose the target using only online L1 observations, NOT evaluation_only.
    observed = [
        (key, value["observation"]["suspicion"])
        for key, value in arm["states"].items()
        if value["observation"] and value["observation"]["valid"]
    ]
    target = max(sorted(observed), key=lambda pair: pair[1])[0]
    assert "Partial.with(DateTimeFieldType,int)" in target
    missing = arm["states"][target]["observation"]["missing_evidence"]
    result = focus_select(graph, target, 4096, ByteCounter(), missing)

    assert result.selector == FOCUSED_SELECTOR
    assert result.status == "ready", (result.reason, result.diagnostics)
    assert result.input_units <= 4096
    assert not result.exact_template_tokens
    assert result.diagnostics["resolved_callees"] == [
        "org.joda.time.base.AbstractPartial.indexOf(DateTimeFieldType)"
    ]
    assert not result.diagnostics["required_missing"]
    assert not result.diagnostics["unresolved_callees"]
    sources = json.loads(result.text)["sources"]
    assert any(
        "if (index == -1)" in unit[2]
        for src in sources for unit in src["units"]
        if src["fn"] == target
    )
    assert any(
        "int index = indexOf(fieldType);" in unit[2]
        for src in sources for unit in src["units"]
    )
    callee = next(src for src in sources if src["fn"].endswith(".indexOf(DateTimeFieldType)"))
    assert "for (int i = 0" in " ".join(unit[2] for unit in callee["units"])
    assert "return -1;" in " ".join(unit[2] for unit in callee["units"])
    assert "BasicDayOfMonthDateTimeField" not in result.text
    for src in sources:
        for start, end, text in src["units"]:
            assert any(n.text == text and n.origin.start_line == start and n.origin.end_line == end
                       and n.owner == src["fn"]
                       for n in graph.nodes)
    checked = package(graph, target, result.ids, 4096, ByteCounter(), selector=result.selector)
    assert checked.text == result.text and checked.input_units == result.input_units


def test_original_bm25_is_still_old_schema():
    graph = Graph.from_dict(json.loads((SNAPSHOT / "graph.json").read_text()))
    target = "org.joda.time.Partial.with(DateTimeFieldType,int)"
    original = relevance_select(graph, target, 4096)
    assert original.selector == "bm25_relevance_only"
    assert json.loads(original.text)["schema"] == "graft-evidence-v1"


def test_saved_graph_offline_audit_never_reads_evaluation_labels():
    from scripts.audit_graft_focused_offline import inspect_instance
    row = inspect_instance(SNAPSHOT, budget=4096)
    assert row["new_llm_calls"] == 0
    assert row["focused_status"] == "ready"
    assert row["focused_input_units"] <= 4096
    assert any(item["method"].endswith(".indexOf(DateTimeFieldType)")
               for item in row["selected_sources"])


def test_new_focused_policy_is_registered_and_separate():
    from rave_fl.graft.runner import GRAFT_POLICIES
    assert "l1_graph_focused" in GRAFT_POLICIES
    assert {"l1_graph", "l1_graph_suspicion", "graft_adaptive"} <= GRAFT_POLICIES


def test_prepare_independent_cohort_requires_existing_manifest(tmp_path):
    from scripts.prepare_graft_focused_independent import prepare
    import pytest

    source = tmp_path / "source.json"
    conf = {
        "benchmark": {
            "bugs": ["Time-1", "Lang-1", "Math-1"],
            "model": "qwen2.5:7b",
            "batch_size": 1,
            "candidate_provider": "flexfl",
            "dataset_file": str(tmp_path / "existing.json"),
        },
        "graft": {"input_budget": 4096, "max_graph": 2, "max_full": 1},
        "graft_policies": ["l1_graph_suspicion"],
    }
    source.write_text(json.dumps(conf))
    with pytest.raises(FileNotFoundError):
        prepare(source, tmp_path / "bad")
    (tmp_path / "existing.json").write_text("{}")
    path = prepare(source, tmp_path / "good")
    out = json.loads(path.read_text())
    assert out["graft_policies"] == [
        "l1_anchor", "l1_graph_suspicion", "l1_graph_focused"
    ]
    assert out["benchmark"]["fusions"] == ["rrf", "suspicion_rank_blend"]
    assert json.loads((path.parent / "provenance.json").read_text())["new_calls_during_preparation"] == 0


def test_focused_arm_runs_in_synthetic_graph_transport(tmp_path):
    from test_graft_controller import setup
    from rave_fl.eval.fusion import Fusion
    from rave_fl.graft.controller import GraftConfig, Router

    candidates, session, store = setup(
        tmp_path, GraftConfig(max_graph=1, max_full=0, input_budget=4096)
    )
    result = Router(candidates, session, Fusion("rrf"), "l1_graph_focused").execute()
    assert not result["budget_exceeded"]
    graph_calls = [call for call in result["calls"] if call["stage"] == "graph"]
    assert graph_calls, result["stop_reason"]
    assert all(c["selection"]["selector"] == FOCUSED_SELECTOR for c in graph_calls)
    assert all(json.loads(c["selection"]["text"])["schema"] == "graft-focused-v1"
               for c in graph_calls)
    assert all(c["selection"]["input_units"] <= 4096 for c in graph_calls)
    assert result["policy"] == "l1_graph_focused"
    assert store.snapshot()["physical_inference_tokens"] == 0
