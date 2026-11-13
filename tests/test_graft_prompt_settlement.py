"""Exercise the approved settlement with paid synthetic responses only."""
import json
from types import SimpleNamespace

import pytest

from rave_fl.core import Action
from rave_fl.eval.backend import ExecutionLimit, PromptSettlement, Store, UnknownUsage
from rave_fl.eval.schema import CompatibilityError, file_hash, fingerprint
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft import prompt_settlement as proof
from rave_fl.graft.fixed_evidence_reporting import audit_saved_run
from test_graft_fixed_evidence_schedule import fixed_fixture


IDENTITY = {"model_manifest_sha256": "synthetic-v1", "serving_version": "graft-fake-v1",
            "binary_sha256": "fixture-binary", "certificate_sha256": "fixture-certificate"}


@pytest.fixture
def store_case(tmp_path):
    requests = {key: {"messages": [{"role": "user", "content": key}],
                      "options": {"num_ctx": 8192, "num_predict": 160}} for key in ("a", "b", "c")}
    bounds = {key: PromptSettlement(fingerprint(value), IDENTITY, 900, 160) for key, value in requests.items()}
    options = {"backend": "fake", "cache_mode": "off", "max_new_calls": 3,
               "physical_budget": 9412, "per_bug_physical_budget": 9412}
    calls = []

    def respond(payload, *_):
        calls.append(payload)
        return {"done": True, "prompt_eval_count": 900, "eval_count": 2, "message": {"content": "{}"}}

    provider = SimpleNamespace(metadata={"digest": "synthetic-v1", "provider_version": "graft-fake-v1",
                                        "prompt_settlement_identity": IDENTITY}, execute=respond)
    actions = {("bug", fixed.T0): (None, None, [(None, None, k, v, 8352, bounds[k]) for k, v in requests.items()])}
    return options, calls, provider, bounds, requests, actions


def test_pending_full_hold_settled_upper_and_paid_invalid_response(tmp_path, store_case):
    opts, calls, provider, bounds, payloads, actions = store_case
    store = Store(tmp_path, opts, settlements=bounds)
    respond = provider.execute

    def inspect(payload, *args):
        assert store.totals()["reserved_or_unknown_tokens"] == 8352
        return respond(payload, *args)

    provider.execute = inspect
    for key in ("a", "b"):
        response, reuse = store.request(key, "bug", provider, payloads[key], [], Action.FULL_VERIFY)
        assert reuse == "fresh" and response["message"]["content"] == "{}"
    assert store.snapshot()["known_provider_tokens"] == 1804
    assert store.snapshot()["settled_upper_tokens"] == 2120
    assert store.snapshot()["reserved_or_unknown_tokens"] == 0
    with pytest.raises(ExecutionLimit, match="cap cannot reserve"):
        store.request("c", "bug", provider, payloads["c"], [], Action.FULL_VERIFY)
    assert len(calls) == 2 and fixed.journal_identity(store, actions)
    store.db.close()


@pytest.mark.parametrize("field", ["physical_budget", "per_bug_physical_budget"])
def test_cap_one_below_serial_boundary(tmp_path, store_case, field):
    opts, calls, provider, bounds, payloads, _ = store_case
    opts[field] -= 1
    store = Store(tmp_path, opts, settlements=bounds)
    store.request("a", "bug", provider, payloads["a"], [], Action.FULL_VERIFY)
    with pytest.raises(ExecutionLimit, match="cap cannot reserve"):
        store.request("b", "bug", provider, payloads["b"], [], Action.FULL_VERIFY)
    assert len(calls) == 1
    store.db.close()


@pytest.mark.parametrize("mode", ["interrupt", "transport", "mismatch", "output_over_cap", "partial"])
def test_unsafe_usage_holds_and_resume_fail_closed(tmp_path, store_case, mode):
    opts, calls, provider, bounds, payloads, actions = store_case
    store = Store(tmp_path, opts, settlements=bounds)
    respond = provider.execute

    def fail(payload, *args):
        response = respond(payload, *args)
        if mode == "interrupt":
            raise KeyboardInterrupt
        if mode == "transport":
            raise TimeoutError
        if mode == "output_over_cap":
            return {**response, "eval_count": 161}
        if mode == "partial":
            return {**response, "done": False}
        return {**response, "prompt_eval_count": 899}

    provider.execute = fail
    error = {"interrupt": KeyboardInterrupt, "transport": TimeoutError, "partial": ValueError}.get(mode, ExecutionLimit)
    with pytest.raises(error):
        store.request("a", "bug", provider, payloads["a"], [], Action.FULL_VERIFY)
    assert store.snapshot()["reserved_or_unknown_tokens"] == 8352
    assert store.snapshot()["settled_upper_tokens"] == 0
    assert store.snapshot()["known_provider_tokens"] == {"mismatch": 901, "output_over_cap": 1061, "partial": 902}.get(mode, 0)
    store.db.close()
    with pytest.raises(CompatibilityError, match="verified request bindings"):
        Store(tmp_path, opts)
    resumed = Store(tmp_path, opts, settlements=bounds)
    with pytest.raises(UnknownUsage, match="unsafe usage"):
        fixed.journal_identity(resumed, actions, reconcile=True)
    with pytest.raises(UnknownUsage):
        resumed.request("b", "bug", provider, payloads["b"], [], Action.FULL_VERIFY)
    assert len(calls) == 1
    resumed.db.close()


def test_crash_after_db_commit_reconciles_once_without_repaying(tmp_path, store_case, monkeypatch):
    opts, calls, provider, bounds, payloads, actions = store_case
    store = Store(tmp_path, opts, settlements=bounds)
    event = store.event

    def crash(record):
        if record["event"] == "response":
            raise KeyboardInterrupt
        event(record)

    with monkeypatch.context() as patch:
        patch.setattr(store, "event", crash)
        with pytest.raises(KeyboardInterrupt):
            store.request("a", "bug", provider, payloads["a"], [], Action.FULL_VERIFY)
    store.db.close()
    resumed = Store(tmp_path, opts, settlements=bounds)
    with pytest.raises(CompatibilityError, match="Missing settlement response"):
        fixed.journal_identity(resumed, actions)
    journal_hash = fixed.journal_identity(resumed, actions, reconcile=True)
    assert journal_hash == fixed.journal_identity(resumed, actions)
    assert resumed.request("a", "bug", provider, payloads["a"], [], Action.FULL_VERIFY)[1] == "journal"
    assert len(calls) == 1
    assert sum("recovered_from_durable_journal" in line for line in resumed.ledger.read_text().splitlines()) == 1
    resumed.db.execute("UPDATE prompt_settlements SET binding='{}'")
    with pytest.raises(CompatibilityError, match="settled usage or provenance"):
        fixed.journal_identity(resumed, actions)
    resumed.db.close()


def test_provenance_mismatch_cannot_create_a_request(tmp_path, store_case):
    opts, calls, provider, bounds, payloads, _ = store_case
    store = Store(tmp_path, opts, settlements=bounds)
    with pytest.raises(CompatibilityError, match="provenance"):
        store.request("a", "bug", provider, payloads["b"], [], Action.FULL_VERIFY)
    provider.metadata["prompt_settlement_identity"] = {**IDENTITY, "binary_sha256": "changed"}
    with pytest.raises(CompatibilityError, match="provenance"):
        store.request("a", "bug", provider, payloads["a"], [], Action.FULL_VERIFY)
    assert not calls and store.snapshot()["new_calls"] == 0
    store.db.close()


def test_exact_token_ids_and_request_hash_are_checked():
    certificate = object.__new__(proof.SettlementCertificate)
    certificate.identity = IDENTITY
    payload = {"options": {"num_predict": 160}, "messages": []}
    certificate.by_action = {("bug", "arm", "target"): {
        "request_sha256": fingerprint(payload), "token_ids_sha256": fingerprint([1, 2]), "prompt_tokens": 2}}
    counter = SimpleNamespace(token_ids=lambda _: [1, 2])
    assert certificate.for_request("bug", "arm", "target", payload, counter).upper == 162
    with pytest.raises(CompatibilityError, match="IDs drifted"):
        certificate.for_request("bug", "arm", "target", payload, SimpleNamespace(token_ids=lambda _: [2, 1]))
    with pytest.raises(CompatibilityError, match="missing from"):
        certificate.for_request("bug", "arm", "target", {**payload, "think": True}, counter)


def test_math_only_certificate_requires_42_math_requests(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}")
    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = tmp_path / name
        path.write_text("{}")
        tokenizer_files[name] = file_hash(path)
    path = tmp_path / "certificate.json"
    value = {
        "schema": proof.SCHEMA, "dataset_sha256": file_hash(dataset), "tokenizer_files": tokenizer_files,
        "serving": {"model_manifest_sha256": proof.MODEL_DIGEST, "template_sha256": proof.TEMPLATE,
                    "serving_version": proof.VERSION, "runner": proof.RUNNER, "system": "Darwin", "architecture": "arm64",
                    **dict.fromkeys(("binary_sha256", "model_blob_sha256", "native_audit_sha256", "native_tokenizer_log_sha256"), "a" * 64)},
        "pairs": {"Math-80": {}},
        "requests": [{"instance_id": "Math-80", "arm": arm, "target": f"m{i}", "prompt_tokens": 900,
                      "request_sha256": "b" * 64, "token_ids_sha256": "c" * 64}
                     for arm in fixed.ARMS for i in range(21)],
    }
    config = {
        "benchmark": {"backend": "ollama", "model": proof.MODEL, "model_digest": proof.MODEL_DIGEST,
                      "context_window": 8192, "output_tokens": 160, "seed": 0, "temperature": 0,
                      "device": "apple", "dataset": "defects4j", "candidate_provider": "flexfl",
                      "candidate_count": 24, "top_per_method": 10, "dataset_file": str(dataset)},
        "graft": {"input_budget": 1024}, "tokenizer_dir": str(tmp_path),
    }

    def load():
        path.write_text(json.dumps(value))
        return proof.SettlementCertificate({**config, "prompt_settlement": {"file": str(path), "sha256": file_hash(path)}})

    certificate = load()
    assert certificate.cohort == ("Math-80",) and certificate.calls == 42
    value["requests"][0]["instance_id"] = "Lang-54"
    with pytest.raises(CompatibilityError, match="audited action"):
        load()
    value["requests"].pop(0)
    with pytest.raises(CompatibilityError, match="entire audited schedule"):
        load()


def test_settlement_flows_through_planner_backend_resume_and_reporting(fixed_fixture, tmp_path, monkeypatch):
    _, config = fixed_fixture
    opts = config["benchmark"]
    baseline, _ = fixed.prepare(config)
    cohort_ids = {p["instance_id"]: len(p["schedule"]) for p in baseline["pairs"]}
    config["prompt_settlement"] = {"file": "fixture", "sha256": "fixture"}

    class FixtureCertificate:
        identity = IDENTITY
        cohort = tuple(cohort_ids)

        def __init__(self, _):
            pass

        def verify_pair(self, *_):
            pass

        def for_request(self, bug, arm, target, payload, counter):
            return PromptSettlement(fingerprint(payload), self.identity, 90, opts["output_tokens"])

    monkeypatch.setattr(fixed, "SettlementCertificate", FixtureCertificate)
    preflight = fixed.GraftProvider.preflight
    execute = fixed.GraftProvider.execute
    calls = []

    def setup(provider):
        metadata = preflight(provider)
        metadata["prompt_settlement_identity"] = IDENTITY
        return metadata

    def respond(provider, payload, ids, action):
        calls.append(payload)
        value = execute(provider, payload, ids, action)
        value.update(prompt_eval_count=90, eval_count=3)
        if len(calls) == 1:
            value["message"]["content"] = "{}"
        return value

    monkeypatch.setattr(fixed.GraftProvider, "preflight", setup)
    monkeypatch.setattr(fixed.GraftProvider, "execute", respond)
    plan, _ = fixed.prepare(config)
    hold = opts["context_window"] + opts["output_tokens"]
    bound = 90 + opts["output_tokens"]
    assert plan["physical_reservation_upper"] == hold + 7 * bound
    assert all(p["physical_reservation_upper"] == hold + 3 * bound for p in plan["pairs"])
    run = tmp_path / "rehearsal"
    result = fixed.run(config, run)
    assert result["spending"]["settled_upper_tokens"] == 8 * bound
    assert result["spending"]["simulated_provider_tokens"] == 8 * 93
    assert result["spending"]["physical_inference_tokens"] == 0
    arm = result["pairs"][0]["arms"][fixed.T0]
    assert not arm["calls"][0]["valid"] and arm["accounting"][0]["settled_upper_tokens"] == bound
    assert arm["accounting"][0]["reserved_tokens"] == hold
    rows, *_ = audit_saved_run(run, None)
    assert rows[0]["planned_settled_upper_tokens"] == 2 * bound
    assert fixed.run(config, run, resume=True)["new_calls_this_invocation"] == 0
    assert len(calls) == 8

    config["per_arm_physical_budget"] = hold + bound - 1
    blocked, _ = fixed.prepare(config)
    assert all(p["status"] == "infeasible" for p in blocked["pairs"])


def test_real_backend_guard_precedes_any_provider(fixed_fixture, tmp_path, monkeypatch):
    _, config = fixed_fixture
    config["benchmark"]["backend"] = "ollama"
    monkeypatch.setattr(fixed, "GraftProvider", lambda *_: pytest.fail("real provider was constructed"))
    with pytest.raises(CompatibilityError, match="attested portable cohort"):
        fixed.run(config, tmp_path / "blocked", allow_inference=True)
    assert not (tmp_path / "blocked").exists()
