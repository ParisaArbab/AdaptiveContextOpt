"""Offline tests for GRAFT Phase-3 test-aware graph retrieval and stopping."""
from pathlib import Path
from types import SimpleNamespace
import json

from rave_fl.graft.graph import Graph
from rave_fl.graft.selection import ByteCounter, package
from rave_fl.graft.test_aware import (
    failure_signal, test_aware_select, stop_decision, SELECTOR,
)
from test_graft_graph import fixture_graph

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "results/graft_phase2_time4_live_20261113_202949/run/instances/Time-4_0d6a675374"


def test_partial_try_fail_is_only_weak_expected_exception_signal():
    snippet = (
        "public void testWith3() {\n"
        "  try {\n    obj.with(value);\n    fail();\n"
        "The last line shown above failed with AssertionFailedError\n"
    )
    result = failure_signal(snippet, "supplied_text:trigger_description")
    assert result["kind"] == "possible_expected_exception"
    assert result["certainty"] == "incomplete_test_excerpt"
    assert result["catch_type"] is None
    assert result["full_test_verified"] is False
    assert result["evidence_origin"] == "supplied_text:trigger_description"

    full = failure_signal(
        "try { call(); fail(); } catch (IllegalArgumentException expected) { }",
        "supplied_text:observed_failure",
    )
    assert full["certainty"] == "catch_present"
    assert full["catch_type"] == "IllegalArgumentException"
    assert full["full_test_verified"] is False
    assert failure_signal("failed junit.framework.AssertionFailedError")["kind"] == "assertion_failure"
    assert failure_signal("an unknown log")["kind"] == "unknown"


def test_conservative_stop_only_when_top1_has_no_open_question():
    obs = SimpleNamespace(
        suspicion=0.92, uncertainty=0.10, valid=True, missing_evidence=""
    )
    assert stop_decision(obs, 1) == "high_suspicion_no_open_evidence_question"
    assert stop_decision(obs, 2) is None
    for missing in ("null check in indexOf() method", "No evidence found"):
        assert stop_decision(SimpleNamespace(
            suspicion=0.92, uncertainty=0.10, valid=True, missing_evidence=missing
        ), 1) is None
    assert stop_decision(SimpleNamespace(
        suspicion=0.83, uncertainty=0.10, valid=True, missing_evidence=""
    ), 1) is None
    assert stop_decision(SimpleNamespace(
        suspicion=0.92, uncertainty=0.21, valid=True, missing_evidence=""
    ), 1) is None
    assert stop_decision(SimpleNamespace(
        suspicion=0.92, uncertainty=0.10, valid=False, missing_evidence=""
    ), 1) is None


def test_time4_test_aware_pack_prioritizes_actual_executable_path():
    graph = Graph.from_dict(json.loads((INSTANCE / "graph.json").read_text()))
    old = json.loads(next(INSTANCE.glob("l1_anchor__rrf_*.json")).read_text())["arm"]
    # Infer target solely from valid online L1 scores. Do not inspect labels.
    scored = sorted(
        (state["observation"]["suspicion"], ident)
        for ident, state in old["states"].items()
        if state.get("observation") and state["observation"].get("valid")
    )
    target = scored[-1][1]
    missing = old["states"][target]["observation"]["missing_evidence"]
    selection = test_aware_select(graph, target, 4096, ByteCounter(), missing)
    assert selection.status == "ready", (selection.reason, selection.diagnostics)
    assert selection.selector == SELECTOR
    assert selection.input_units <= 4096
    info = selection.diagnostics
    assert info["test_signal"]["kind"] == "possible_expected_exception"
    assert info["test_signal"]["certainty"] == "incomplete_test_excerpt"
    assert not info["test_signal"]["full_test_verified"]
    assert info["test_signal"]["evidence_origin"] == "supplied_text:trigger_description"
    assert info["gold_used"] is False
    assert not info["runtime_reachability_proven"]
    assert "indexOf" in info["requested_calls"]
    # The new policy prioritizes full target path fragments over the
    # model's speculative callee hint.
    payload = json.loads(selection.text)
    assert payload["schema"] == "graft-test-aware-v1"
    assert payload["test_signal"] == info["test_signal"]
    blocks = [
        unit[2] for section in payload["sources"]
        if section["fn"] == target
        for unit in section["units"]
    ]
    assert any("if (index == -1)" in x for x in blocks)
    assert any("iChronology.validate(" in x for x in blocks)
    assert any("getField(index).set(" in x for x in blocks)
    assert any("return new Partial(this, newValues)" in x for x in blocks)
    # Same source fragments, unchanged, within the same counting contract.
    for section in payload["sources"]:
        for start, end, src in section["units"]:
            assert any(
                n.owner == section["fn"]
                and n.origin.start_line == start
                and n.origin.end_line == end
                and n.text == src
                for n in graph.nodes
            )
    reconstructed = package(
        graph, target, selection.ids, 4096, ByteCounter(),
        selector=SELECTOR,
    )
    assert reconstructed.text == selection.text
    assert reconstructed.ids == selection.ids
    assert reconstructed.input_units == selection.input_units


def test_small_budget_keeps_original_sources_and_reports_missing():
    _, _, graph = fixture_graph()
    target = "x.A.a()"
    selection = test_aware_select(
        graph, target, 2000, ByteCounter(), missing_evidence="unknownCall()",
    )
    assert selection.status in {"ready", "infeasible"}
    assert selection.diagnostics["unresolved_callees"]
    assert selection.diagnostics["semantic_completeness_guaranteed"] is False
    assert selection.diagnostics["gold_used"] is False
    if selection.status == "ready":
        assert selection.input_units <= 2000


def test_router_treatments_coexist_on_fake_provider(tmp_path):
    from test_graft_controller import setup
    from rave_fl.graft.controller import GraftConfig, Router
    from rave_fl.eval.fusion import Fusion
    from rave_fl.graft.runner import GRAFT_POLICIES

    assert {"l1_graph_test_aware", "l1_graph_focused", "l1_graph_suspicion"} <= GRAFT_POLICIES
    candidates, session, store = setup(
        tmp_path, GraftConfig(max_graph=1, max_full=0)
    )
    arm = Router(candidates, session, Fusion("rrf"), "l1_graph_test_aware").execute()
    assert arm["policy"] == "l1_graph_test_aware"
    assert not arm["budget_exceeded"]
    assert all(c["stage"] != "method_full" for c in arm["calls"])
    for c in arm["calls"]:
        if c["stage"] == "graph":
            assert c["selection"]["selector"] == SELECTOR
            assert json.loads(c["selection"]["text"])["schema"] == "graft-test-aware-v1"
    assert store.snapshot()["physical_inference_tokens"] == 0


def test_stop_rule_changes_router_actions_not_just_helper(tmp_path):
    from dataclasses import replace as dc_replace
    from test_graft_controller import setup
    from rave_fl.eval.fusion import Fusion
    from rave_fl.graft.controller import GraftConfig, Router, EvidenceStage
    from rave_fl.pilot_types import CandidateState

    class FixedSketchRouter(Router):
        def observe(self, candidates, stage, selection=None):
            super().observe(candidates, stage, selection)
            if stage == EvidenceStage.SKETCH:
                for candidate in candidates:
                    state = self.states[candidate.function_id]
                    previous = state.observation
                    self.states[candidate.function_id] = CandidateState(
                        level=1,
                        observation=dc_replace(
                            previous,
                            suspicion=0.95 if candidate.rank == 1 else 0.10,
                            uncertainty=0.05 if candidate.rank == 1 else 0.75,
                            missing_evidence="",
                        ),
                        history=state.history,
                    )

    def run(policy, directory):
        cs, session, _ = setup(
            directory, GraftConfig(max_graph=1, max_full=0, input_budget=4096)
        )
        result = FixedSketchRouter(
            cs, session, Fusion("suspicion_rank_blend"), policy
        ).execute()
        options = next(
            event for event in result["trace"]
            if event["phase"] == "route" and event.get("selected")
        )
        return cs, result, options

    cs, gated, gated_route = run(
        "l1_graph_test_aware", tmp_path / "with_stop"
    )
    _, ungated, ungated_route = run(
        "l1_graph_test_aware_nostop", tmp_path / "without_stop"
    )
    top = next(c.function_id for c in cs if c.rank == 1)
    assert any(
        row["candidate"] == top and
        row["reason"] == "high_suspicion_no_open_evidence_question"
        for row in gated_route["options"]
    )
    assert gated_route["selected"]["candidate"] != top
    assert ungated_route["selected"]["candidate"] == top
    assert not gated["budget_exceeded"] and not ungated["budget_exceeded"]
