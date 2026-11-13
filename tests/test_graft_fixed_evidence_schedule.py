"""Provider-free contracts for the fixed propagation comparison."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rave_fl.eval.adapters import load_instances, read_rows
from rave_fl.eval.fixtures import create
from rave_fl.eval.runner import prepare_one
from rave_fl.eval.schema import CompatibilityError, safe_name
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.graph import build_graph
from rave_fl.pilot import write_json
from test_graft_topology import synthetic_bundle


@pytest.fixture
def fixed_fixture(tmp_path):
    root = tmp_path / "fixture"
    original = create(root)
    rows, _ = read_rows(original)
    labels = [{"instance_id": row["instance_id"], "gold": row.pop("gold")} for row in rows]
    write_json(original, {"version": "synthetic-v1", "instances": rows})
    labels_file = root / "labels.json"
    write_json(labels_file, {"instances": labels})
    bundles = root / "embeddings"
    value = {
        "benchmark": {
            "dataset": "custom",
            "dataset_file": str(original),
            "backend": "fake",
            "model": "synthetic-fixture",
            "context_window": 4096,
            "budget": 20000,
            "per_bug_physical_budget": 20000,
            "physical_budget": 40000,
        },
        "input_allowance": 2048,
        "embeddings_dir": str(bundles),
        "labels_file": str(labels_file),
    }
    config = fixed.configuration(value)
    opts = config["benchmark"]
    for instance in load_instances(opts)[0]:
        instance, entities, *_ = prepare_one(instance, opts)
        graph = build_graph(instance, entities)
        write_json(
            bundles / (safe_name(instance.instance_id) + ".json"), synthetic_bundle(graph).value
        )
    return value, config


def test_dry_run_is_byte_stable_and_provider_free(fixed_fixture, tmp_path, monkeypatch):
    value, _ = fixed_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run opened a provider, journal or labels")

    monkeypatch.setattr(fixed, "GraftProvider", forbidden)
    monkeypatch.setattr(fixed, "Store", forbidden)
    monkeypatch.setattr(fixed, "load_gold", forbidden)
    original = fixed.read_rows

    def online_only(path):
        assert str(path) != value["labels_file"]
        return original(path)

    monkeypatch.setattr(fixed, "read_rows", online_only)
    path = tmp_path / "config.json"
    write_json(path, value)
    outputs = [tmp_path / "plan-a.json", tmp_path / "plan-b.json"]
    for output in outputs:
        assert fixed.main(["dry-run", "--config", str(path), "--output", str(output)]) == 0
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    plan = json.loads(outputs[0].read_text())
    assert plan["new_model_calls"] == 0
    assert plan["planned_fresh_calls_upper"] == 8
    assert plan["arm_diff"] == {"propagate": {"T0": False, "T1": True}}
    a, b = (plan["arm_configs"][key] for key in (fixed.T0, fixed.T1))
    assert {k for k in a if a[k] != b[k]} == {"propagate"}
    assert plan["config"]["graft"]["paths"] is plan["config"]["graft"]["repair"] is False
    for pair in plan["pairs"]:
        assert pair["graph_sha256"] and pair["embeddings_sha256"] and pair["pool_sha256"]
        targets = [item["target"] for item in pair["schedule"]]
        for name in (fixed.T0, fixed.T1):
            assert [r["target"] for r in pair["arms"][name]] == targets
        for a, b in zip(pair["arms"][fixed.T0], pair["arms"][fixed.T1], strict=True):
            assert a["counter"] == b["counter"]
            assert a["input_units"] <= value["input_allowance"]
            assert b["input_units"] <= value["input_allowance"]
            for row in (a, b):
                assert (
                    json.loads(row["request"]["messages"][1]["content"])["target"] == row["target"]
                )
            assert a["request"]["messages"][0] == b["request"]["messages"][0]
            assert a["request"]["options"] == b["request"]["options"]
            assert {k for k in a["request"] if a["request"][k] != b["request"][k]} <= {"messages"}


def test_schedule_preserves_ties_and_rejects_duplicate_or_missing_targets():
    a, b, c = [SimpleNamespace(function_id=k, rank=r) for k, r in [("a", 2), ("b", 2), ("c", 1)]]
    entities = [SimpleNamespace(entity_id=k) for k in ("a", "b", "c")]
    assert fixed.fixed_schedule([b, a, c], entities) == [c, b, a]
    for candidates, pool in [
        ([a, b], entities),
        ([a, a, c], entities),
        ([a, b, c], [entities[0], entities[0], entities[2]]),
    ]:
        with pytest.raises(CompatibilityError, match="unique prepared entity"):
            fixed.fixed_schedule(candidates, pool)


def test_plan_rejects_a_second_changed_selector_flag(fixed_fixture, monkeypatch):
    _, config = fixed_fixture
    monkeypatch.setitem(fixed.ARMS, fixed.T1, replace(fixed.ARMS[fixed.T1], typed=True))
    with pytest.raises(CompatibilityError, match="only in propagation"):
        fixed.prepare(config)


def test_offline_labels_are_rejected_in_online_manifest(fixed_fixture):
    _, config = fixed_fixture
    path = Path(config["benchmark"]["dataset_file"])
    data = json.loads(path.read_text())
    data["instances"][0]["gold"] = {"functions": ["private-label"]}
    write_json(path, data)
    with pytest.raises(CompatibilityError, match="Remove labels"):
        fixed.prepare(config)


@pytest.mark.parametrize("allow", [False, True])
def test_uncertified_run_cannot_instantiate_live_provider(fixed_fixture, tmp_path, monkeypatch, allow):
    value, _ = fixed_fixture
    value["benchmark"]["backend"] = "ollama"
    config = fixed.configuration(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("Real provider instantiated")

    monkeypatch.setattr(fixed, "GraftProvider", forbidden)
    output = tmp_path / "no-live"
    with pytest.raises(
        CompatibilityError, match="attested portable cohort" if allow else "--allow-inference"
    ):
        fixed.run(config, output, allow_inference=allow)
    assert not output.exists()


def test_source_mismatch_keeps_selected_bug_incomplete(fixed_fixture, tmp_path):
    value, _ = fixed_fixture
    value["benchmark"]["bugs"] = ["synthetic-java"]
    config = fixed.configuration(value)
    path = Path(value["benchmark"]["dataset_file"]).parent / "checkout" / "Example.java"
    path.write_text(path.read_text() + "// a tracked source change\n")
    plan, _ = fixed.prepare(config)
    assert len(plan["pairs"]) == 1
    assert plan["pairs"][0]["status"] == "infeasible"
    assert "Tracked checkout files modified" in plan["pairs"][0]["reasons"][0]
    result = fixed.run(config, tmp_path / "mismatch")
    assert result["pairs"][0]["pair_status"] == "incomplete"
    assert result["new_calls_this_invocation"] == 0
