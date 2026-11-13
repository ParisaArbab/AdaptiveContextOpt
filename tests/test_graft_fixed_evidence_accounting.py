"""Exercise paired completion and the real journal with synthetic responses."""
import json
import sqlite3

import pytest

from rave_fl.eval.schema import CompatibilityError
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.controller import Router
from rave_fl.pilot import write_json
from test_graft_fixed_evidence_schedule import fixed_fixture


def test_fixed_collection_seals_labels_and_reconciles_every_paid_call(
    fixed_fixture, tmp_path, monkeypatch
):
    value, config = fixed_fixture
    executed = []
    execute = fixed.GraftProvider.execute
    load_gold = fixed.load_gold
    read_rows = fixed.read_rows

    def call(provider, payload, ids, action):
        assert "unseen-evaluation-marker" not in json.dumps(payload)
        executed.extend(ids)
        return execute(provider, payload, ids, action)

    def check_labels(instance, source):
        assert len(executed) == 8
        return load_gold(instance, source)

    def check_reads(path):
        if str(path) == value["labels_file"]:
            assert len(executed) == 8
        return read_rows(path)

    def router_forbidden(*args, **kwargs):
        raise AssertionError("Router selected an inspection")

    labels = read_rows(value["labels_file"])[0]
    labels[0]["gold"]["semantics"] = "unseen-evaluation-marker"
    write_json(tmp_path / "fixture" / "labels.json", {"instances": labels})
    monkeypatch.setattr(fixed.GraftProvider, "execute", call)
    monkeypatch.setattr(Router, "execute", router_forbidden)
    monkeypatch.setattr(fixed, "load_gold", check_labels)
    monkeypatch.setattr(fixed, "read_rows", check_reads)
    output = tmp_path / "run"
    result = fixed.run(config, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert "unseen-evaluation-marker" not in json.dumps(manifest)
    assert result["new_calls_this_invocation"] == 8
    assert result["spending"]["physical_inference_tokens"] == 0
    paid = 0
    expected_order = []
    for pair, record in zip(result["pairs"], manifest["plan"]["pairs"], strict=True):
        assert pair["pair_status"] == "complete"
        targets = [item["target"] for item in record["schedule"]]
        for name in (fixed.T0, fixed.T1):
            arm = pair["arms"][name]
            expected_order += targets
            assert [call["target"] for call in arm["calls"]] == targets
            assert set(arm["ranked_functions"]) == set(targets)
            actual = sum(c["input_tokens"] + c["output_tokens"] for c in arm["calls"])
            assert actual == arm["logical_tokens"]
            assert actual == sum(e["actual_tokens"] for e in arm["accounting"])
            paid += actual
    assert executed == expected_order
    assert paid == result["spending"]["simulated_provider_tokens"]
    resumed = fixed.run(config, output, resume=True)
    assert resumed == {**result, "new_calls_this_invocation": 0}
    assert len(executed) == 8


def test_infeasible_target_blocks_both_arms_without_shortening_schedule(
    fixed_fixture, tmp_path, monkeypatch
):
    from dataclasses import replace

    _, config = fixed_fixture
    select = fixed.topo_select

    def one_infeasible(graph, target, bundle, budget, *, counter, config, audit):
        selection = select(graph, target, bundle, budget, counter=counter, config=config, audit=audit)
        return (
            replace(selection, status="infeasible", reason="fixture")
            if config.propagate
            else selection
        )

    monkeypatch.setattr(fixed, "topo_select", one_infeasible)
    plan, _ = fixed.prepare(config)
    for pair in plan["pairs"]:
        assert len(pair["schedule"]) == 2
        assert all(row["status"] == "ready" for row in pair["arms"][fixed.T0])
        assert all(row["status"] == "infeasible" for row in pair["arms"][fixed.T1])
    result = fixed.run(config, tmp_path / "infeasible")
    assert result["spending"]["new_calls"] == 0
    assert all(p["pair_status"] == "incomplete" for p in result["pairs"])
    assert all(a["ranked_functions"] is None for p in result["pairs"] for a in p["arms"].values())


def test_call_ceiling_blocks_collection_before_charging(fixed_fixture, tmp_path):
    value, _ = fixed_fixture
    value["per_arm_max_new_calls"] = 1
    result = fixed.run(fixed.configuration(value), tmp_path / "limit")
    assert result["new_calls_this_invocation"] == 0
    assert all("call limit" in " ".join(p["reasons"]) for p in result["pairs"])


def test_invalid_verdict_stays_paid_and_excludes_pair(fixed_fixture, tmp_path, monkeypatch):
    value, _ = fixed_fixture
    value["benchmark"]["bugs"] = ["synthetic-java"]
    execute = fixed.GraftProvider.execute
    count = 0

    def invalid_once(provider, payload, ids, action):
        nonlocal count
        response = execute(provider, payload, ids, action)
        count += 1
        if count == 1:
            response["message"]["content"] = "{}"
        return response

    monkeypatch.setattr(fixed.GraftProvider, "execute", invalid_once)
    result = fixed.run(fixed.configuration(value), tmp_path / "invalid")
    pair = result["pairs"][0]
    assert pair["pair_status"] == "incomplete"
    assert pair["arms"][fixed.T0]["execution_status"] == "incomplete"
    assert pair["arms"][fixed.T1]["execution_status"] == "complete"
    assert all(
        a["paired_status"] == "incomplete" and "metrics" not in a for a in pair["arms"].values()
    )
    paid_invalid = pair["arms"][fixed.T0]["calls"][0]
    assert not paid_invalid["valid"] and paid_invalid["input_tokens"] > 0
    assert count == result["spending"]["new_calls"] == 4


def test_interrupted_resume_reuses_paid_journal_row(fixed_fixture, tmp_path, monkeypatch):
    _, config = fixed_fixture
    output = tmp_path / "interrupted"
    execute = fixed.FixedGraphBackend.execute

    def stop_after_payment(backend, candidates, action, level):
        execute(backend, candidates, action, level)
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(fixed.FixedGraphBackend, "execute", stop_after_payment)
        with pytest.raises(KeyboardInterrupt):
            fixed.run(config, output)
    assert json.loads((output / "accounting.json").read_text())["new_calls"] == 1
    result = fixed.run(config, output, resume=True)
    assert result["new_calls_this_invocation"] == 7
    assert result["spending"]["new_calls"] == 8
    assert all(p["pair_status"] == "complete" for p in result["pairs"])
    first = result["pairs"][0]["arms"][fixed.T0]
    assert first["accounting"][0]["reuse"] == "journal"
    assert first["logical_tokens"] == sum(
        c["input_tokens"] + c["output_tokens"] for c in first["calls"]
    )


def test_unknown_usage_is_never_retried(fixed_fixture, tmp_path, monkeypatch):
    _, config = fixed_fixture
    output = tmp_path / "unknown"
    calls = 0

    def timeout(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("synthetic transport timeout")

    monkeypatch.setattr(fixed.GraftProvider, "execute", timeout)
    with pytest.raises(TimeoutError):
        fixed.run(config, output)
    with pytest.raises(fixed.UnknownUsage, match="unsafe usage"):
        fixed.run(config, output, resume=True)
    spending = json.loads((output / "accounting.json").read_text())
    assert spending["usage_unknown"] and spending["reserved_or_unknown_tokens"] == 4256
    assert calls == spending["new_calls"] == 1


def test_resume_refuses_changed_inputs_or_deleted_journal_rows(fixed_fixture, tmp_path):
    _, config = fixed_fixture
    output = tmp_path / "tampered"
    fixed.run(config, output)
    original = config["graft"]["input_budget"]
    config["graft"]["input_budget"] = original + 1
    with pytest.raises(CompatibilityError, match="manifest differs"):
        fixed.run(config, output, resume=True)
    config["graft"]["input_budget"] = original
    with sqlite3.connect(output / "journal.sqlite3") as db:
        db.execute("DELETE FROM calls WHERE rowid=(SELECT min(rowid) FROM calls)")
    with pytest.raises(CompatibilityError, match="ledger disagree"):
        fixed.run(config, output, resume=True)
