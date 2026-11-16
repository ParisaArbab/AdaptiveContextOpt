"""Real archived Defects4J evidence, no inference or test executions.

The experiments retain full failure nodes and exact source evidence; the new
bounded view is line-numbered and sha256 verifiable, and only opt-in.
"""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from rave_fl.graft.failure_excerpt import bounded_failure_excerpt
from rave_fl.graft.focused import focus_select
from rave_fl.graft.graph import Graph
from rave_fl.graft.selection import ByteCounter, mandatory_ids, package
from rave_fl.graft.test_aware import test_aware_select
from scripts.verify_graft_phase3_graph_feasibility import inspect


ARCHIVE = Path("results/graft_phase3_real_bugs_20261115_132745/run")
BUGS = {
    "Lang-1": "Lang-1_ad630962a9",
    "Math-1": "Math-1_76b556abc3",
    "Time-1": "Time-1_8f01918918",
}


def graph_for(bug):
    folder = ARCHIVE / "instances" / BUGS[bug]
    return Graph.from_dict(json.loads((folder / "graph.json").read_text()))


@pytest.mark.parametrize("bug", ["Lang-1", "Math-1", "Time-1"])
def test_reconstructs_exact_original_failure_lines_from_real_bug(bug):
    graph = graph_for(bug)
    full = next(n.text for n in graph.nodes if n.kind == "failure")
    view = bounded_failure_excerpt(full)
    assert view["source_sha256"] == sha256(full.encode()).hexdigest()
    assert view["excerpt_is_incomplete"] is True
    assert view["original_utf8_bytes"] > 4096
    assert all(full.splitlines()[i-1] == text for i, text in view["selected_original_lines"])
    assert sum(len(text.encode()) + 1 for _, text in view["selected_original_lines"]) <= 1650
    assert "EXECUTED BUGGY TEST" in str(view["selected_original_lines"])
    assert any(text.startswith("--- ") for _, text in view["selected_original_lines"])
    assert len(view["selected_original_lines"]) < view["original_line_count"]


@pytest.mark.parametrize("bug", ["Lang-1", "Math-1", "Time-1"])
def test_bounded_focused_and_test_aware_mandatory_are_ready(bug):
    graph = graph_for(bug)
    target = next(iter(graph.targets))
    for choose, selector in [
        (focus_select, "focused_bounded_failure_v2"),
        (test_aware_select, "test_aware_bounded_failure_v2"),
    ]:
        selected = choose(
            graph, target, 4096, ByteCounter(), "", failure_digest_mode="bounded_excerpt"
        )
        assert selected.selector == selector
        assert set(mandatory_ids(graph, target)) <= set(selected.ids)
        assert selected.status == "ready", (bug, selector, selected.input_units, selected.reason)
        assert selected.input_units <= 4096
        payload = json.loads(selected.text)
        assert payload["failure_excerpt"]["excerpt_is_incomplete"]
        assert "failure" not in payload
        assert payload["failure_excerpt"]["source_sha256"] == sha256(
            next(n.text for n in graph.nodes if n.kind == "failure").encode()
        ).hexdigest()
        checked = package(
            graph, target, selected.ids, 4096, ByteCounter(), selector=selector
        )
        assert checked.text == selected.text
        assert checked.status == "ready"


def test_old_graph_view_retains_entire_failure():
    graph = graph_for("Math-1")
    target = next(iter(graph.targets))
    original = package(
        graph, target, mandatory_ids(graph, target), 4096,
        ByteCounter(), selector="test_aware_compact_v1"
    )
    assert original.status == "infeasible"
    old = json.loads(original.text)
    assert old["failure"] == next(n.text for n in graph.nodes if n.kind == "failure")
    assert "failure_excerpt" not in old
    assert old["schema"] == "graft-test-aware-v1"


def test_unknown_failure_structure_fails_closed():
    with pytest.raises(ValueError, match="header layout"):
        bounded_failure_excerpt("Just some log text")
    with pytest.raises(ValueError, match="observed"):
        bounded_failure_excerpt(
            "OFFICIAL DEFECTS4J BUG: X\n"
            "EXECUTED BUGGY TEST: T\n"
            "TEST SOURCE: F\n"
            "TEST SOURCE SHA256: " + "a" * 64 + "\n"
            "COMPLETE EXECUTED TEST METHOD (verbatim):\n"
            "void t() {}\n"
            "OBSERVED DEFECTS4J FAILING_TESTS OUTPUT (verbatim):\n"
        )


def test_real_online_l1_data_can_prepare_graph_before_inference():
    config = {
        "graft": {"input_budget": 4096, "failure_digest_mode": "bounded_excerpt"}
    }
    readiness = inspect(ARCHIVE, config)
    assert readiness["new_model_calls"] == 0
    assert readiness["uses_saved_online_l1_not_gold"] is True
    assert readiness["ready_for_real_graph_comparison"] is True, [
        (row["bug"], row["policy"], row["ready_count_of_24"],
         row["ready_in_previous_online_l1_top5"])
        for row in readiness["rows"]
    ]
