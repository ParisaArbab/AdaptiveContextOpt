"""Provider-free input relocation and readiness checks."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest

from rave_fl.eval.schema import CompatibilityError, Evidence, fingerprint
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.fixed_inputs import fixed_graph
from rave_fl.graft.portable_experiment import condition, shard_record
from rave_fl.pilot import write_json
from scripts import audit_graft_phase6 as audit
from test_graft_fixed_evidence_schedule import fixed_fixture
from test_graft_graph import fixture_graph


def file_backed(value):
    dataset = Path(value["benchmark"]["dataset_file"])
    manifest = json.loads(dataset.read_text())
    for row in manifest["instances"]:
        source = row.pop("evidence")
        name = row["instance_id"] + ".log"
        (dataset.parent / name).write_text(source["text"], encoding="utf-8")
        row["evidence_file"] = name
    write_json(dataset, manifest)
    return dataset


def test_file_backed_evidence_cohort_survives_relocation(fixed_fixture, tmp_path):
    value, _ = fixed_fixture
    dataset = file_backed(value)
    other = tmp_path / "second mount"
    shutil.copytree(dataset.parent, other)
    relocated = {
        **value, "benchmark": {**value["benchmark"], "dataset_file": str(other / dataset.name)},
        "embeddings_dir": str(other / "embeddings"), "labels_file": str(other / "labels.json"),
    }
    original, _ = fixed.prepare(fixed.configuration(value))
    moved, _ = fixed.prepare(fixed.configuration(relocated))
    assert all(pair["status"] == "ready" for pair in original["pairs"] + moved["pairs"])
    assert original != moved
    assert original["pairs"][0]["prepared_sha256"] != moved["pairs"][0]["prepared_sha256"]
    assert shard_record(original) == shard_record(moved)
    assert condition(original, "same-labels", None) == condition(moved, "same-labels", None)
    for pair in moved["pairs"]:
        assert any(n["origin"]["path"].startswith("artifact:sha256:") for d in pair["evidence_comparisons"]
                   for n in d["evidence_nodes"] if n["kind"] == "failure")
    archive = moved["pairs"][0]["graph_sha256"]
    (other / (moved["pairs"][0]["instance_id"] + ".log")).write_text("Different exact evidence")
    changed, _ = fixed.prepare(fixed.configuration(relocated))
    assert changed["pairs"][0]["status"] == "infeasible"
    assert changed["pairs"][0].get("graph_sha256") != archive


def test_absolute_path_in_payload_is_not_silently_rewritten():
    instance, entities, _ = fixture_graph()
    evidence = Evidence("observed_failure", "Failure in /mnt/project/A.java:3", "/logs/failed")
    with pytest.raises(CompatibilityError, match="local resource path"):
        fixed_graph(replace(instance, checkout="/mnt/project", evidence=evidence), entities)


def test_different_model_dataset_and_sample_are_config_conditions(fixed_fixture):
    value, _ = fixed_fixture
    baseline, _ = fixed.prepare(fixed.configuration(value))
    changed = {**value, "benchmark": {**value["benchmark"], "model": "second-model-fixture",
                                      "dataset": "defects4j", "bugs": ["synthetic-java"],
                                      "limit": 1, "device": "a100"}}
    second, _ = fixed.prepare(fixed.configuration(changed))
    assert second["new_model_calls"] == 0 and len(second["pairs"]) == 1
    assert second["pairs"][0]["status"] == "ready"
    assert second["arm_configs"] == baseline["arm_configs"]
    assert second["arm_diff"] == baseline["arm_diff"]
    assert condition(second, None, None) != condition(baseline, None, None)
    assert {row["request"]["model"] for arm in second["pairs"][0]["arms"].values()
            for row in arm} == {"second-model-fixture"}


def test_pinned_git_blob_ignores_worktree_rewrite(tmp_path):
    repo = tmp_path / "metadata"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    path = repo / "revision.csv"
    path.write_text("pinned\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
                    "commit", "-qm", "archive"], check=True)
    head = audit.git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    path.write_text("edited\n")
    output = tmp_path / "verified"
    assert audit.local_blob(repo, head, "revision.csv", output)["status"] == "verified_pinned_archive"
    assert output.read_text() == "pinned\n"
    assert audit.local_blob(repo, head, "missing", output)["status"] == "absent_from_tree"


def test_gguf_metadata_reader_leaves_tensor_bytes_unread(tmp_path):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    header += string("tokenizer.ggml.pre") + struct.pack("<I", 8) + string("llama-bpe")
    header += string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + string("a") + string("b")
    path = tmp_path / "fixture.gguf"
    path.write_bytes(header + b"unused tensor")
    metadata = audit.gguf_tokenizer_metadata(path)
    assert metadata["metadata_bytes_read"] == len(header)
    assert metadata["tokenizer.ggml.tokens.entries"] == 2
    assert "tokenizer.ggml.tokens" not in metadata
    vocabulary = audit.gguf_tokenizer_metadata(path, retain_vocab=True)
    assert vocabulary["tokenizer.ggml.tokens"] == ["a", "b"]
    assert vocabulary["metadata_bytes_read"] == len(header)
    path.write_bytes(header[:-1])
    with pytest.raises(CompatibilityError, match="offline read bound"):
        audit.gguf_tokenizer_metadata(path)


def test_exact_token_reference_compares_ids_and_preserves_provider_count(tmp_path, monkeypatch):
    request = {"model": "fixture", "messages": [{"role": "user", "content": "fixture"}]}
    plan = {"config": {"benchmark": {"model_digest": "model-sha"}}, "pairs": [{
        "status": "ready", "arms": {fixed.T0: [{"request": request}], fixed.T1: [{"request": request}]},
    }]}
    server = {"model_manifest_sha256": "model-sha", "ollama_template_sha256": "template-sha"}
    reference = {**server, "serving_build": "synthetic-test", "capture_kind": "offline_serving_tokenizer",
                 "requests": [{"request_sha256": fingerprint(request), "token_ids": [1, 2, 3],
                               "provider_prompt_eval_count": 2}]}
    files = {key: tmp_path / (key + ".json") for key in ("plan", "audit", "reference")}
    for key, data in (("plan", plan), ("audit", server), ("reference", reference)):
        write_json(files[key], data)
    counter = SimpleNamespace(token_ids=lambda messages: [1, 2, 3], name="fixture")
    monkeypatch.setattr(audit, "TokenizerCounter", lambda path: counter)
    for expected, ids in ((0, [1, 2, 3]), (3, [1, 9, 3])):
        reference["requests"][0]["token_ids"] = ids
        write_json(files["reference"], reference)
        output = tmp_path / f"compare-{expected}"
        args = SimpleNamespace(**files, tokenizer=tmp_path, output=output)
        assert audit.compare(args) == expected
        result = json.loads((output / "reference-comparison.json").read_text())
        row, = result["requests"]
        assert row["local_template_tokens"] == row["reference_token_ids_count"] == 3
        assert row["provider_prompt_eval_count"] == 2 and row["identical_ids"] == (expected == 0)
        assert result["admission_bound_changed"] is False
