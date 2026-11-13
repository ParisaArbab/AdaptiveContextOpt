"""Offline prompt, budget and accounting checks for the fixed T0/T1 runner."""

from dataclasses import replace
import json

import pytest

from rave_fl.core import approx_tokens
from rave_fl.eval.schema import CodeEntity, CompatibilityError, Evidence, fingerprint
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.graph import build_graph
from rave_fl.graft.selection import ByteCounter, SYSTEM, mandatory_ids, package
from test_graft_fixed_evidence_schedule import fixed_fixture
from test_graft_graph import fixture_graph


def test_local_template_counts_complete_messages_but_claims_no_serving_parity(fixed_fixture, tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    local = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    local.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    path = tmp_path / "tokenizer"
    local.save_pretrained(path)
    value, _ = fixed_fixture
    value["tokenizer_dir"] = str(path)
    plan, _ = fixed.prepare(fixed.configuration(value))
    assert plan["new_model_calls"] == 0
    for pair in plan["pairs"]:
        assert pair["status"] == "ready"
        for a, b in zip(pair["arms"][fixed.T0], pair["arms"][fixed.T1], strict=True):
            assert a["input_allowance"] == b["input_allowance"] == 2048
            assert a["request"]["options"] == b["request"]["options"]
            for row in (a, b):
                messages = row["request"]["messages"]
                user = messages[1]["content"]
                audit = row["prompt_accounting"]
                assert messages[0] == {"role": "system", "content": SYSTEM}
                assert audit["messages_sha256"] == fingerprint(messages)
                assert audit["local_chat_template_tokens"] == len(f"system {SYSTEM} user {user} assistant".split())
                assert row["input_units"] == audit["local_chat_template_tokens"]
                assert audit["serving_template_tokens"] is None
                assert audit["byte_proxy_units"] == len(SYSTEM.encode()) + len(user.encode()) + 256
                assert audit["generation_output_cap_tokens"] == row["request"]["options"]["num_predict"]


@pytest.mark.parametrize("body", ["pass", 'return "é猫"', 'return "' + "猫" * 2000 + '"'])
def test_budget_boundary_utf8_and_duplicate_anchors(body):
    instance, _, _ = fixture_graph()
    instance = replace(instance, language="python", evidence=Evidence("observed_failure", "é猫 failure"))
    target = CodeEntity("x.py::f", "x.py", 1, 2, f"def f():\n    {body}\n", 1.0)
    graph = build_graph(instance, [target])
    anchors = mandatory_ids(graph, target.entity_id)
    ids = tuple(n.id for n in graph.nodes)
    selection = package(graph, target.entity_id, ids + anchors + anchors, 10000)
    assert len(selection.ids) == len(ids)
    assert json.loads(selection.text)["failure"] == "é猫 failure"
    budget = len(SYSTEM.encode()) + len(selection.text.encode()) + 256
    assert selection.input_units == budget > len(SYSTEM) + len(selection.text) + 256
    assert package(graph, target.entity_id, ids, budget, ByteCounter()).status == "ready"
    assert package(graph, target.entity_id, ids, budget - 1).status == "infeasible"
    assert package(graph, target.entity_id, ids, 1).status == "infeasible"
    with pytest.raises(CompatibilityError):
        replace(target, source="  \n")


def tight_limits(value):
    """Use the exact serial peak from an independently reconstructed fixture schedule."""
    plan, _ = fixed.prepare(fixed.configuration(value))
    pairs = plan["pairs"]
    hold = value["benchmark"]["context_window"] + plan["config"]["benchmark"]["output_tokens"]

    def bounds(rows):
        lengths = [r["logical_reservation"] for r in rows]
        assert max(lengths) < hold
        return sum(lengths[:-1]) + hold

    groups = [[pair["arms"][name] for name in (fixed.T0, fixed.T1)] for pair in pairs]
    value["per_arm_physical_budget"] = max(bounds(arm) for group in groups for arm in group)
    value["benchmark"]["per_bug_physical_budget"] = max(bounds(a + b) for a, b in groups)
    value["benchmark"]["physical_budget"] = bounds([r for group in groups for arm in group for r in arm])
    return hold


def test_tight_serial_caps_admit_complete_pair_and_resume(fixed_fixture, tmp_path, monkeypatch):
    value, _ = fixed_fixture
    hold = tight_limits(value)
    config = fixed.configuration(value)
    plan, _ = fixed.prepare(config)
    assert plan["run_admitted"] and all(p["status"] == "ready" for p in plan["pairs"])
    assert plan["physical_envelope"] == {
        "serial_peak_tokens": value["benchmark"]["physical_budget"],
        "sum_full_window_reservations_tokens": 8 * hold,
    }
    assert 8 * hold > value["benchmark"]["physical_budget"]
    execute = fixed.FixedGraphBackend.execute

    def interrupt(backend, candidates, action, level):
        execute(backend, candidates, action, level)
        raise KeyboardInterrupt

    out = tmp_path / "tight"
    with monkeypatch.context() as patch:
        patch.setattr(fixed.FixedGraphBackend, "execute", interrupt)
        with pytest.raises(KeyboardInterrupt):
            fixed.run(config, out)
    result = fixed.run(config, out, resume=True)
    assert result["new_calls_this_invocation"] == 7
    assert result["spending"]["new_calls"] == 8
    assert result["spending"]["physical_inference_tokens"] == 0
    assert result["spending"]["reserved_or_unknown_tokens"] == 0
    for pair, record in zip(result["pairs"], plan["pairs"], strict=True):
        assert pair["pair_status"] == "complete"
        a, b = (pair["arms"][name] for name in (fixed.T0, fixed.T1))
        for name, arm in pair["arms"].items():
            counts = [max(1, approx_tokens("".join(m["content"] for m in row["request"]["messages"])))
                      for row in record["arms"][name]]
            assert arm["provider_reported_input_tokens"] == sum(counts)
            assert arm["simulated_physical_tokens"] == arm["logical_tokens"]
        delta = pair["paired_differences"]
        assert delta["usage"]["logical_tokens"] == b["logical_tokens"] - a["logical_tokens"]
        assert delta["accuracy"]["legacy_v1"]["method"]["MRR"] == (
            b["metrics"]["legacy_v1"]["method"]["MRR"] - a["metrics"]["legacy_v1"]["method"]["MRR"]
        )
    assert fixed.run(config, out, resume=True)["new_calls_this_invocation"] == 0


@pytest.mark.parametrize("scope", ["arm", "bug", "run"])
def test_one_token_below_envelope_fails_at_its_scope(fixed_fixture, scope):
    value, _ = fixed_fixture
    hold = tight_limits(value)
    if scope == "arm":
        value["per_arm_physical_budget"] -= 1
    else:
        field = "per_bug_physical_budget" if scope == "bug" else "physical_budget"
        value["benchmark"][field] -= 1
    plan, _ = fixed.prepare(fixed.configuration(value))
    expected = {"arm": "per-arm physical", "bug": "per-bug physical", "run": "whole-run token"}[scope]
    assert any(expected in reason for pair in plan["pairs"] for reason in pair["reasons"])
    assert plan["new_model_calls"] == 0
    full = [{"logical_reservation": hold, "prompt_accounting": {"physical_reservation_tokens": hold}}] * 4
    assert fixed.physical_envelope(full)["serial_peak_tokens"] == 4 * hold


@pytest.mark.parametrize("message", [None, {"content": {}}, {"content": '{"suspicion":' + "1" * 400 + '}'}])
def test_invalid_transport_message_is_charged_and_kept_in_ledger(fixed_fixture, tmp_path, monkeypatch, message):
    value, _ = fixed_fixture
    value["benchmark"]["bugs"] = ["synthetic-java"]
    execute = fixed.GraftProvider.execute
    calls = 0

    def send(provider, payload, ids, action):
        nonlocal calls
        response = execute(provider, payload, ids, action)
        calls += 1
        if calls == 1:
            response["message"] = message
        return response

    monkeypatch.setattr(fixed.GraftProvider, "execute", send)
    path = tmp_path / "invalid"
    result = fixed.run(fixed.configuration(value), path)
    pair, = result["pairs"]
    call = pair["arms"][fixed.T0]["calls"][0]
    assert not call["valid"] and call["input_tokens"] > 0 and call["output_tokens"] > 0
    assert pair["pair_status"] == "incomplete" and pair["paired_differences"]["accuracy"] is None
    assert pair["paired_differences"]["usage"] is not None
    events = [json.loads(line) for line in (path / "raw_ledgers/requests.jsonl").read_text().splitlines()]
    reply = next(e for e in events if e["event"] == "response")
    assert reply["response"]["message"] == message
    assert calls == result["spending"]["new_calls"] == 4


def test_resume_checks_settled_prefix_usage_after_interruption(fixed_fixture, tmp_path, monkeypatch):
    _, config = fixed_fixture
    original_send = fixed.GraftProvider.execute
    original_request = fixed.Store.request

    def excessive_input(provider, payload, ids, action):
        response = original_send(provider, payload, ids, action)
        response["prompt_eval_count"] = payload["options"]["num_ctx"]
        return response

    def interrupt(store, *args, **kwargs):
        original_request(store, *args, **kwargs)
        raise KeyboardInterrupt

    path = tmp_path / "settled-before-validation"
    with monkeypatch.context() as patch:
        patch.setattr(fixed.GraftProvider, "execute", excessive_input)
        patch.setattr(fixed.Store, "request", interrupt)
        with pytest.raises(KeyboardInterrupt):
            fixed.run(config, path)
    with pytest.raises(CompatibilityError, match="accounting differs"):
        fixed.run(config, path, resume=True)
    usage = json.loads((path / "accounting.json").read_text())
    assert usage["new_calls"] == 1 and usage["simulated_provider_tokens"] > 4096


def test_interrupt_before_response_keeps_full_hold_and_blocks_retry(fixed_fixture, tmp_path, monkeypatch):
    _, config = fixed_fixture
    attempts = 0

    def interrupt(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(fixed.GraftProvider, "execute", interrupt)
    path = tmp_path / "pending"
    with pytest.raises(KeyboardInterrupt):
        fixed.run(config, path)
    with pytest.raises(fixed.UnknownUsage, match="pending"):
        fixed.run(config, path, resume=True)
    usage = json.loads((path / "accounting.json").read_text())
    assert usage["pending"] and usage["reserved_or_unknown_tokens"] == 4256
    assert usage["known_provider_tokens"] == 0 and attempts == 1


def test_resume_requires_chronological_request_prefix(fixed_fixture, tmp_path):
    _, config = fixed_fixture
    path = tmp_path / "out-of-order"
    fixed.run(config, path)
    ledger = path / "raw_ledgers/requests.jsonl"
    lines = ledger.read_text().splitlines(keepends=True)
    first, second = [i for i, line in enumerate(lines) if json.loads(line)["event"] == "request"][:2]
    lines[first], lines[second] = lines[second], lines[first]
    ledger.write_text("".join(lines))
    with pytest.raises(CompatibilityError, match="prefix of the fixed schedule"):
        fixed.run(config, path, resume=True)
