"""Offline tests for the A100 deployment and its failure boundaries."""
from collections import Counter
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rave_fl.core import Action
from rave_fl.eval.backend import ExecutionLimit, Store, UnknownUsage
from rave_fl.eval.schema import CompatibilityError
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.ollama_live import AttestedOllama
from scripts.audit_graft_schedule_tokens import linux_ids
from scripts.graft_server import batch, data, selection
from test_graft_prompt_settlement import store_case


def test_frozen_cohort_is_nested_disjoint_and_excludes_development():
    manifest = selection.read_selection()
    assert manifest["sha256"] == "251767e96283e3855ac1d7389db1d76c015ce8551c213a5acb4aafa0efd0c2b6"
    ids = [[r["defect"] for r in manifest[segment]] for segment in ("core", "extension")]
    assert [len(Counter(b.split("-")[0] for b in group)) for group in ids] == [14, 14]
    assert not set(ids[0]) & set(ids[1])
    assert not {"Lang-54", "Math-80", "Closure-61", "JxPath-14"} & set(ids[0] + ids[1])
    for row in manifest["core"] + manifest["extension"]:
        assert set(row["rankings"]) == set(selection.RANKERS)


def test_entrypoints_only_allow_explicit_inference(tmp_path):
    scripts = selection.ROOT / "scripts/graft_server"
    for command in ("prepare_linux", "prepare_batch", "run_batch", "report_batch"):
        entry = scripts / (command + ".sh")
        result = subprocess.run([entry, "--help"], capture_output=True,
                                env={**os.environ, "GRAFT_PYTHON": sys.executable}, check=True)
        assert b"usage:" in result.stdout
    denied = subprocess.run([scripts / "run_batch.sh", "--count", "100", "--work", tmp_path / "missing",
                             "--host-config", tmp_path / "also-missing"], capture_output=True,
                            env={**os.environ, "GRAFT_PYTHON": sys.executable})
    assert denied.returncode == 3
    assert b"Explicit --allow-inference required" in denied.stderr
    assert not list(tmp_path.iterdir())


def test_preserve_incompatible_checksum(tmp_path):
    path = tmp_path / "manifest.json"
    batch.sealed(path, {"members": ["A", "B"]})
    assert batch.unseal(path)["members"] == ["A", "B"]
    path.write_text(path.read_text().replace('"A"', '"X"'))
    with pytest.raises(CompatibilityError, match="Corrupt sealed manifest"):
        batch.unseal(path)


@pytest.mark.parametrize("bad_smoke", [True, False])
def test_batch_smoke_resume_and_incomplete_followup(tmp_path, monkeypatch, bad_smoke):
    groups = [{"id": name, "members": [bug], "config": "config", "lock": "lock",
               "calls": 2, "output": str(tmp_path / name)}
              for name, bug in (("core-smoke-Gson-9", "Gson-9"), ("core-000", "Math-1"), ("core-001", "Math-2"))]
    for group in groups[:2]:
        Path(group["output"]).mkdir()
    (Path(groups[0]["output"]) / "results.json").touch()
    dispatch = []

    def mock_portable(_, command, *flags):
        output = Path(flags[flags.index("--output") + 1])
        output.mkdir(exist_ok=command == "resume")
        (output / "results.json").touch()
        dispatch.append((command, output.name))
        if output.name == "core-000":
            raise subprocess.CalledProcessError(3, "fixture")

    def mock_audit(g):
        complete = g["id"] == groups[2]["id"] or g["id"] == groups[0]["id"] and not bad_smoke
        return ([{"pair_status": "complete" if complete else "incomplete"}] * 2,
                None, None, None, {"new_calls": 2}, None)

    monkeypatch.setattr(batch, "implementation_identity", lambda: {})
    monkeypatch.setattr(batch, "deployment_identity", lambda: "frozen")
    monkeypatch.setattr(batch, "portable", mock_portable)
    monkeypatch.setattr(batch, "audit_group", mock_audit)
    args = SimpleNamespace(work=tmp_path, allow_inference=True, binary="b", manifest="m", models=tmp_path / "models")
    plan = {"groups": groups, "implementation": {}, "deployment_sha256": "frozen", "selection_sha256": "fixture"}
    if bad_smoke:
        with pytest.raises(CompatibilityError, match="smoke failed"):
            batch.execute(args, [plan])
        assert dispatch == []
    else:
        batch.execute(args, [plan])
        assert dispatch == [("resume", "core-000"), ("run", "core-001")]
        batch.execute(args, [plan])
        assert len(dispatch) == 2
        args.work = tmp_path / "other"
        with pytest.raises(CompatibilityError, match="bound to another workspace"):
            batch.execute(args, [plan])


def test_unstarted_reports_keep_dispositions_without_accuracy(tmp_path):
    args = SimpleNamespace(work=tmp_path, output=tmp_path / "report")
    batch.report_batch(args, [{"dispositions": [{"defect": "A", "status": "BLOCKED", "reason": "bad span"},
                                              {"defect": "B", "status": "READY", "group": "test"}],
                               "groups": [{"id": "test", "output": str(tmp_path / "run"), "members": ["B"]}],
                               "max_calls": 96, "aggregate_physical_cap": 144000}])
    summary = json.loads((args.output / "batch-summary.json").read_text())
    assert summary["selected"] == 2 and summary["pair_complete_labeled"] == 0
    assert not (args.output / "matched-results").exists()
    rows = list(csv.DictReader((args.output / "disposition.csv").read_text().splitlines()))
    assert [r["execution_status"] for r in rows] == ["not_prepared", "unstarted"]


def test_retry_recertifies_only_blocked_defects_before_dispatch(tmp_path, monkeypatch):
    selection_manifest = {"core": [{"defect": "A"}, {"defect": "B"}], "sha256": "selection",
                          "label_reference_offline_only": {}}
    calls = []
    monkeypatch.setattr(batch, "implementation_identity", lambda: {})
    monkeypatch.setattr(batch, "deployment_identity", lambda: "deploy")
    monkeypatch.setattr(batch, "git", lambda *_: b"commit")
    monkeypatch.setattr(batch, "rank_blob", lambda *_: None)
    batch.write_json(tmp_path / "offline/ground-truth.json", {})

    def prepare(_, case, *unused):
        calls.append(case["defect"])
        return {**case, "status": "PREPARED" if case["defect"] == "A" or calls.count("B") == 2 else "BLOCKED"}

    def certify(_, path, members):
        ids = [row["defect"] for row in members]
        return {"id": path.name, "members": ids, "calls": 2 * len(ids), "peak": 100,
                "offline_recall": {ident: {"status": "missing_labels"} for ident in ids}}

    monkeypatch.setattr(batch, "prepare_one_defect", prepare)
    monkeypatch.setattr(batch, "certify", certify)
    args = SimpleNamespace(work=tmp_path, workers=1, flexfl=None, retry_blocked=False)
    original = batch.stage(args, selection_manifest, "core")
    args.retry_blocked = True
    updated = batch.stage(args, selection_manifest, "core")
    assert calls == ["A", "B", "B"]
    assert updated["selected"] == original["selected"]
    assert updated["groups"][0] == original["groups"][0]
    assert updated["groups"][1]["members"] == ["B"]
    assert batch.unseal(tmp_path / "core/batch-preparation-r1.json") == original


def test_pinned_blob_and_archived_failure_checks(tmp_path, monkeypatch):
    raw = b"pinned ranking"
    spec = {"git_blob": data.blob_sha(raw), "path": "rank.csv"}
    path = tmp_path / "rank.csv"
    monkeypatch.setattr(data, "git", lambda *_: raw)
    assert data.rank_blob(spec, path, tmp_path)["bytes"] == len(raw)
    path.write_bytes(b"drift")
    with pytest.raises(CompatibilityError, match="differs"):
        data.rank_blob(spec, path, tmp_path)
    archived = "--- p.Test::check\njunit.framework.AssertionFailedError\n  at p.Test.check(Test.java:3)\n"
    target, body = data.archive_blocks(archived)[0]
    assert target == "p.Test::check" and data.failure_signature(body)[2] == ["at p.Test.check(Test.java:3)"]
    with pytest.raises(CompatibilityError, match="lacks project"):
        data.failure_signature(archived.replace("p.Test.check(Test.java:3)", "java.base/Foo.bar(Foo.java:1)"))


def test_coordinates_keep_order_and_reject_extra_code(tmp_path):
    source = tmp_path / "source/src/main/java/p/C.java"
    source.parent.mkdir(parents=True)
    source.write_text("package p;\nclass C {\n int field;\n // comment\n int f() { return 1; }\n}\n")
    checkout = tmp_path / "checkout"
    checkout.symlink_to(tmp_path / "source", target_is_directory=True)
    rankings = tmp_path / "ranks"
    for ranker in selection.RANKERS:
        ranking = rankings / f"FlexFL/data/FL_results/{ranker}/Defects4J/Lang-1000_method-susps.csv"
        ranking.parent.mkdir(parents=True)
        ranking.write_text("File,Signature,StartLine,EndLine\np.C,f(),4,5\n")
    valid = data.audit_pool(checkout, rankings, "Lang-1000")
    assert not valid["failures"] and valid["ordered_candidates"] == ["p.C.f()"]
    first = rankings / "FlexFL/data/FL_results/SBIR/Defects4J/Lang-1000_method-susps.csv"
    first.write_text(first.read_text().replace("4,5", "3,5"))
    drift = data.audit_pool(checkout, rankings, "Lang-1000")
    assert drift["ordered_candidates"] == valid["ordered_candidates"]
    assert drift["failures"] == ["p.C.f()"]
    assert drift["rows"][0]["start_line"] == 3 and "unrelated noncomment" in drift["rows"][0]["reason"]


def test_native_tokenizer_checks_rendered_bytes_and_receipt(tmp_path):
    binary = tmp_path / "mock"
    binary.write_text("#!/bin/sh\ncat >/dev/null\nprintf 'vocab only - skipping tensors\\n' >&2\n"
                      "printf '%s\\n' '{\"rendered\":\"rendered\",\"token_ids\":[128000,1]}'\n")
    binary.chmod(0o700)
    messages = {b"rendered": [], b"different": []}
    assert list(linux_ids([b"rendered"], messages, binary, "model", "template", tmp_path)) == [[128000, 1]]
    with pytest.raises(CompatibilityError, match="template differs"):
        list(linux_ids([b"different"], messages, binary, "model", "template", tmp_path))


def test_linux_listener_can_restart_with_only_process_fields_changed():
    provider = object.__new__(AttestedOllama)
    provider.a100_gpus = [{"uuid": "A"}]
    provider.metadata = {"name": "model", "listener": {"pid": 5, "started": "now", "inode": 3}, "a100_inventory": ["A"]}
    original = {**provider.metadata, "listener": {"pid": 4, "started": "then", "inode": 3}}
    assert provider.matches_cohort(original)
    assert not provider.matches_cohort({**original, "listener": {**original["listener"], "inode": 99}})
    assert not provider.matches_cohort({**original, "a100_inventory": ["B"]})


@pytest.mark.parametrize("gpu_ok", [True, False])
def test_linux_paid_device_check_and_durable_reconciliation(tmp_path, store_case, monkeypatch, gpu_ok):
    opts, calls, provider, bounds, payloads, _ = store_case
    identity = {**provider.metadata["prompt_settlement_identity"], "system": "Linux"}
    provider.metadata["prompt_settlement_identity"] = identity
    bounds = {k: replace(v, serving_identity=identity) for k, v in bounds.items()}
    actions = {("bug", fixed.T0): (None, None, [(None, None, k, v, 8352, bounds[k]) for k, v in payloads.items()])}
    receipt = {"model_size": 10, "size_vram": 10, "context_length": 8192,
               "compute_processes": [{"pid": 3, "gpu_uuid": "A", "allocated_mib": 12}], "listener": {"pid": 2}}
    provider.verify_generation_device = lambda: receipt if gpu_ok else None
    store = Store(tmp_path, opts, settlements=bounds)
    event = store.event

    def crash(value):
        if value["event"] == "response" and gpu_ok:
            raise KeyboardInterrupt
        event(value)

    monkeypatch.setattr(store, "event", crash)
    with pytest.raises(KeyboardInterrupt if gpu_ok else ExecutionLimit):
        store.request("a", "bug", provider, payloads["a"], [], Action.FULL_VERIFY)
    assert store.snapshot()["known_provider_tokens"] == 902
    assert store.snapshot()["reserved_or_unknown_tokens"] == (0 if gpu_ok else 8352)
    store.db.close()
    resumed = Store(tmp_path, opts, settlements=bounds)
    if gpu_ok:
        assert fixed.journal_identity(resumed, actions, reconcile=True)
        assert resumed.request("a", "bug", provider, payloads["a"], [], Action.FULL_VERIFY)[1] == "journal"
        events = [json.loads(line) for line in resumed.ledger.read_text().splitlines()]
        assert [e["gpu_attestation"] for e in events if e["event"] == "response"] == [receipt]
    else:
        with pytest.raises(UnknownUsage):
            fixed.journal_identity(resumed, actions, reconcile=True)
    assert len(calls) == 1
    resumed.db.close()
