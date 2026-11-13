"""Exercise the real transport and collector with loopback fixture responses."""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import threading
from types import SimpleNamespace

import pytest

from rave_fl.eval.backend import PromptSettlement
from rave_fl.eval.schema import CompatibilityError, fingerprint
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft import ollama_live as live
from rave_fl.graft import portable_experiment as portable
from rave_fl.graft.fixed_evidence_reporting import LIVE_LABEL, report
from rave_fl.graft.prompt_settlement import MODEL, MODEL_DIGEST
from rave_fl.pilot import write_json
from test_graft_fixed_evidence_schedule import fixed_fixture


HOST = "local_mac" if platform.system() == "Darwin" else "remote_linux_a100"
TEMPLATE = "fixture no-tools template"


@pytest.fixture
def server(tmp_path, monkeypatch, fixed_fixture):
    value, config = fixed_fixture
    baseline, _ = fixed.prepare(config)
    sizes = {pair["instance_id"]: len(pair["schedule"]) for pair in baseline["pairs"]}
    state = SimpleNamespace(posts=[], metadata=[], version="0.17.7", model_digest=MODEL_DIGEST,
                            template=TEMPLATE, mode="valid", pid=123)
    model_root = tmp_path / "models"
    manifest = model_root / "manifests/registry.ollama.ai/library/llama3.1/8b"
    binary = tmp_path / "ollama"
    for path in (manifest, binary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    blobs = tuple(model_root / name for name in ("model", "template", "config"))
    for path in blobs:
        path.write_text("fixture")
    identity = {"model_manifest_sha256": MODEL_DIGEST, "serving_version": "0.17.7",
                "template_sha256": sha256(TEMPLATE.encode()).hexdigest(), "certificate_sha256": "fixture"}

    class Certificate:
        cohort = tuple(sizes)
        calls = 2 * sum(sizes.values())

        def __init__(self, config):
            self.identity = identity

        def verify_live_caps(self, config):
            pass

        def local_model_paths(self, *paths):
            return blobs

        def verify_local_host(self, *paths):
            pass

        def verify_pair(self, *_):
            pass

        def for_request(self, bug, arm, target, payload, counter):
            return PromptSettlement(fingerprint(payload), identity, 90, payload["options"]["num_predict"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def respond(self, payload=None):
            if self.path == "/api/chat":
                state.posts.append(payload)
                if state.mode == "disconnect":
                    return
                verdict = {"suspicion": 0.7, "uncertainty": 0.2, "explanation": "fixture", "missing_evidence": ""}
                content = "{}" if state.mode == "invalid_once" and len(state.posts) == 1 else json.dumps(verdict)
                body = {"done": True, "model": "wrong:8b" if state.mode == "wrong_model" else MODEL,
                        "prompt_eval_count": 91 if state.mode == "count_drift" else 90,
                        "eval_count": 5, "message": {"content": content}}
                if state.mode == "missing_usage":
                    del body["eval_count"]
            else:
                state.metadata.append(self.path)
                body = {"/api/version": {"version": state.version},
                        "/api/tags": {"models": [{"name": MODEL, "digest": state.model_digest}]},
                        "/api/show": {"template": state.template, "modelfile": f"FROM {blobs[0]}",
                                      "model_info": {"general.architecture": "llama"},
                                      "capabilities": ["completion", "tools"]}}[self.path]
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    # The local fixture supplies only attestation primitives and response bytes.
    # Its requests go through the production launcher, runner, transport and store.
    monkeypatch.setattr(live, "listener", lambda *_: {"pid": state.pid})
    for module in (live, fixed, portable):
        monkeypatch.setattr(module, "SettlementCertificate", Certificate)
    original_resources = fixed.resources

    def fixture_resources(cfg, bug):
        return original_resources({**cfg, "benchmark": {**cfg["benchmark"], "backend": "fake"}}, bug)

    monkeypatch.setattr(fixed, "resources", fixture_resources)
    monkeypatch.setattr(portable, "optional_model_check", lambda *_: {})
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            value["benchmark"].update(backend="ollama", model=MODEL, model_digest=MODEL_DIGEST,
                                      bugs=list(sizes), endpoint=f"http://127.0.0.1:{httpd.server_port}",
                                      context_window=8192, budget=36000, per_bug_physical_budget=72000,
                                      physical_budget=216000, max_new_calls=144)
            value["prompt_settlement"] = {"file": "fixture", "sha256": "a" * 64}
            config_path = tmp_path / "live-config.json"
            write_json(config_path, value)
            # Loopback fixture has no physical GPU, regardless of the test host.
            flags = ["--profile", HOST, "--device", "auto", "--config", str(config_path), "--serving-binary", str(binary),
                     "--generation-manifest", str(manifest), "--ollama-models-dir", str(model_root),
                     "--endpoint", value["benchmark"]["endpoint"]]
            yield state, flags
        finally:
            httpd.shutdown()
            thread.join()


def freeze(tmp_path, flags):
    preflight = tmp_path / "preflight"
    assert portable.main(["preflight", *flags, "--attest-live", "--output", str(preflight)]) == 0
    diag = json.loads((preflight / "preflight.json").read_text())
    assert diag["status"] == "ready_for_manual_live" and diag["planned_calls"] == 8 and diag["new_model_calls"] == 0
    plan = json.loads((preflight / "plans/shard-000.json").read_text())
    payloads = [row["request"] for pair in plan["pairs"] for arm in pair["arms"].values() for row in arm]
    return ["--cohort", str(preflight / "cohort.json")], payloads


def test_manual_flag_exact_requests_paid_invalid_and_sealed_report(server, tmp_path):
    state, flags = server
    cohort, payloads = freeze(tmp_path, flags)
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as blocked:
        portable.main(["run", *flags, *cohort, "--output", str(out)])
    assert blocked.value.code == 2 and not state.posts and not out.exists()
    state.mode = "invalid_once"
    run = [*flags, *cohort, "--allow-inference", "--output", str(out)]
    assert portable.main(["run", *run]) == 3
    assert state.posts == payloads
    result = json.loads((out / "results.json").read_text())
    assert result["synthetic"] is False and result["pairs"][0]["pair_status"] == "incomplete"
    assert result["spending"]["physical_inference_tokens"] == 8 * 95
    assert result["spending"]["settled_upper_tokens"] == 8 * 250
    assert result["spending"]["simulated_provider_tokens"] == 0
    assert portable.main(["resume", *run]) == 3 and len(state.posts) == 8
    summary = report(out, tmp_path / "fixture-report")
    assert summary["label"] == LIVE_LABEL and summary["spending"] == result["spending"]
    assert summary["selected_pairs"] == 2
    with pytest.raises(SystemExit) as blocked:
        portable.main(["run", *flags, *cohort, "--allow-inference", "--output", str(tmp_path / "duplicate")])
    assert blocked.value.code == 2 and len(state.posts) == 8


@pytest.mark.parametrize("mode,actual", [("wrong_model", 95), ("count_drift", 96),
                                         ("disconnect", 0), ("missing_usage", 0)])
def test_paid_or_unknown_live_fault_stops_and_retains_hold(server, tmp_path, mode, actual):
    state, flags = server
    cohort, _ = freeze(tmp_path, flags)
    state.mode = mode
    out = tmp_path / "fault"
    run = [*flags, *cohort, "--allow-inference", "--output", str(out)]
    for subcommand in ("run", "resume"):
        with pytest.raises(SystemExit) as blocked:
            portable.main([subcommand, *run])
        assert blocked.value.code == 2
        accounting = json.loads((out / "accounting.json").read_text())
        assert accounting["known_provider_tokens"] == actual
        assert accounting["reserved_or_unknown_tokens"] == 8352
        assert accounting["settled_upper_tokens"] == 0 and accounting["new_calls"] == 1
        assert not (out / "results.json").exists() and len(state.posts) == 1


@pytest.mark.parametrize("field,value", [("model_digest", "foreign"), ("version", "0.12.2"),
                                         ("template", "foreign"), ("pid", 124)])
def test_endpoint_drift_after_cohort_cannot_dispatch(server, tmp_path, field, value):
    state, flags = server
    cohort, _ = freeze(tmp_path, flags)
    setattr(state, field, value)
    out = tmp_path / "unverified"
    with pytest.raises(SystemExit) as blocked:
        portable.main(["run", *flags, *cohort, "--allow-inference", "--output", str(out)])
    assert blocked.value.code == 2 and not state.posts and not out.exists()


def test_known_paid_prefix_resumes_over_loopback(server, tmp_path, monkeypatch):
    state, flags = server
    cohort, payloads = freeze(tmp_path, flags)
    out = tmp_path / "interrupt"
    run = [*flags, *cohort, "--allow-inference", "--output", str(out)]
    original = fixed.FixedGraphBackend.execute

    def stop_after_payment(*args):
        original(*args)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(fixed.FixedGraphBackend, "execute", stop_after_payment)
        with pytest.raises(KeyboardInterrupt):
            portable.main(["run", *run])
    assert len(state.posts) == 1
    assert portable.main(["resume", *run]) == 0
    assert state.posts == payloads


def test_listener_rejects_non_loopback_before_inspection(tmp_path):
    with pytest.raises(CompatibilityError, match="direct"):
        live.AttestedOllama({"endpoint": "http://localhost:11434", "model": MODEL, "backend": "ollama"},
                           None, tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize("fault", ["wildcard", "owner", "inode", "runner", "store"])
def test_os_listener_evidence_fails_closed(tmp_path, monkeypatch, fault):
    if platform.system() != "Darwin":
        pytest.skip("Fixture exercises the Darwin executable vnode branch")
    binary = tmp_path / "ollama"
    binary.write_text("fake executable")
    stat = binary.stat()
    models = tmp_path / "models"

    def lsof(*args):
        if args[0].startswith("-iTCP:"):
            return [{"pid": 12, "n": "*:1234" if fault == "wildcard" else "127.0.0.1:1234"}]
        return [{"n": str(binary), "D": hex(stat.st_dev), "i": str(stat.st_ino + (fault == "inode"))}]

    monkeypatch.setattr(live, "lsof_records", lsof)
    monkeypatch.setattr(live.subprocess, "check_output", lambda *a, **kw: f"{os.getuid() + (fault == 'owner')} start")
    monkeypatch.setattr(live, "process_arguments", lambda *_: (
        [os.fsencode(binary), b"serve"], [
            b"OLLAMA_MODELS=" + os.fsencode(tmp_path if fault == "store" else models),
            b"OLLAMA_NEW_ENGINE=" + (b"true" if fault == "runner" else b"false"),
        ]))
    with pytest.raises(CompatibilityError):
        live.listener(1234, binary, models)


def test_metadata_change_between_requests_stops_before_next_dispatch(server, tmp_path, monkeypatch):
    state, flags = server
    cohort, _ = freeze(tmp_path, flags)
    out = tmp_path / "drift"
    execute = fixed.FixedGraphBackend.execute

    def drift(*args):
        response = execute(*args)
        state.template = "changed after the paid response"
        return response

    monkeypatch.setattr(fixed.FixedGraphBackend, "execute", drift)
    with pytest.raises(SystemExit) as blocked:
        portable.main(["run", *flags, *cohort, "--allow-inference", "--output", str(out)])
    assert blocked.value.code == 2 and len(state.posts) == 1
    usage = json.loads((out / "accounting.json").read_text())
    assert usage["known_provider_tokens"] == 95 and usage["reserved_or_unknown_tokens"] == 0


def test_smoke_needs_its_own_schedule_certificate(server, tmp_path):
    state, flags = server
    out = tmp_path / "wrong-schedule"
    assert portable.main(["preflight", *flags, "--attest-live", "--limit", "1", "--output", str(out)]) == 3
    assert not (out / "cohort.json").exists() and not state.posts
    assert "selected cohort differs" in (out / "preflight.json").read_text()
