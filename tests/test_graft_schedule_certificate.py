"""Offline checks for new immutable live cohort certificates."""
import json
from types import SimpleNamespace

import pytest

from rave_fl.eval.schema import CompatibilityError, file_hash, fingerprint
from rave_fl.graft import prompt_settlement as proof


@pytest.fixture
def schedule_case(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}")
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = tmp_path / name
        path.write_text("{}")
        files[name] = file_hash(path)
    encoder = {"checkpoint_revision": proof.BGE_REVISION, "checkpoint_files": {"fixture": "a" * 64},
               "max_length": 8192, "batch_size": 8, "device_requested": "cpu",
               "packages": {"FlagEmbedding": "1.4.2", "transformers": "4.56.1", "torch": "2.8.0"}}
    monkeypatch.setattr(proof, "BGE_FILE_MAP_SHA256", fingerprint(encoder["checkpoint_files"]))
    schedule = [{"target": target, "prior_rank": i} for i, target in enumerate(("a", "b"), 1)]
    payload = {"messages": [], "options": {"num_predict": 160}}
    value = {"schema": proof.SCHEDULE_SCHEMA, "dataset_sha256": file_hash(dataset), "tokenizer_files": files,
             "serving": {"model_manifest_sha256": proof.MODEL_DIGEST, "template_sha256": proof.TEMPLATE,
                         "serving_version": proof.VERSION, "runner": proof.RUNNER,
                         "system": "Darwin", "architecture": "arm64", "binary_sha256": proof.MAC_BINARY_SHA256,
                         **dict.fromkeys(("model_blob_sha256", "native_audit_sha256", "native_tokenizer_log_sha256",
                                          "capture_plan_sha256"), "b" * 64)},
             "encoder": encoder, "schedules": {"Project-123": ["a", "b"]},
             "pairs": {"Project-123": {"pool_sha256": "c" * 64, "graph_sha256": "d" * 64,
                                       "embeddings_sha256": "e" * 64, "schedule_sha256": fingerprint(schedule)}},
             "requests": [{"instance_id": "Project-123", "arm": arm, "target": target, "prompt_tokens": 2,
                           "request_sha256": fingerprint(payload), "token_ids_sha256": fingerprint([128000, 11])}
                          for arm in proof.ARMS for target in ("a", "b")]}
    cfg = {"benchmark": {"backend": "ollama", "model": proof.MODEL, "model_digest": proof.MODEL_DIGEST,
                         "context_window": 8192, "output_tokens": 160, "seed": 0, "temperature": 0,
                         "device": "apple", "dataset": "defects4j", "candidate_provider": "flexfl",
                         "candidate_count": 24, "top_per_method": 10, "dataset_file": str(dataset),
                         "bugs": ["Project-123"], **proof.CAPS},
           "graft": {"input_budget": 1024}, "tokenizer_dir": str(tmp_path), **proof.ARM_CAPS}
    path = tmp_path / "certificate.json"

    def load():
        path.write_text(json.dumps(value))
        return proof.SettlementCertificate({**cfg, "prompt_settlement": {"file": str(path), "sha256": file_hash(path)}})

    return load, value, cfg, schedule, payload


def test_new_schedule_binds_targets_encoder_and_exact_prompt(schedule_case):
    load, value, _, schedule, payload = schedule_case
    cert = load()
    assert cert.calls == 4 and cert.cohort == ("Project-123",)
    graph = SimpleNamespace(hash="d" * 64)
    bundle = SimpleNamespace(hash="e" * 64, value={"provenance": value["encoder"]},
                             model_id=proof.BGE_MODEL, dimension=1024, synthetic=False)
    cert.verify_pair("Project-123", {"pool_hash": "c" * 64}, graph, bundle, schedule)
    counter = SimpleNamespace(token_ids=lambda _: [128000, 11])
    assert cert.for_request("Project-123", proof.ARMS[0], "a", payload, counter).upper == 162
    with pytest.raises(CompatibilityError, match="Candidate order"):
        cert.verify_pair("Project-123", {"pool_hash": "c" * 64}, graph, bundle, list(reversed(schedule)))
    bundle.synthetic = True
    with pytest.raises(CompatibilityError, match="Embedding provenance"):
        cert.verify_pair("Project-123", {"pool_hash": "c" * 64}, graph, bundle, schedule)


@pytest.mark.parametrize("change, message", [
    ("scope", "exact explicit cohort"), ("cap", "original approved caps"), ("host", "exact explicit cohort"),
    ("missing", "entire audited schedule"), ("order", "immutable paired schedules"), ("encoder", "frozen BGE-M3"),
])
def test_new_schedule_rejects_drift(schedule_case, change, message):
    load, value, cfg, *_ = schedule_case
    if change == "scope":
        cfg["benchmark"]["bugs"] = ["Other-1"]
    elif change == "cap":
        cfg["benchmark"]["budget"] += 1
    elif change == "host":
        value["serving"]["system"] = "Linux"
    elif change == "missing":
        value["requests"].pop()
    elif change == "order":
        value["requests"][0], value["requests"][1] = value["requests"][1], value["requests"][0]
    else:
        value["encoder"]["checkpoint_files"]["fixture"] = "f" * 64
    with pytest.raises(CompatibilityError, match=message):
        load()
