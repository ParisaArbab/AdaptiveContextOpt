"""Exact Qwen cache-seed tests. All data synthetic, no Ollama/Defects4J calls."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from rave_fl.eval.schema import CompatibilityError, fingerprint
from scripts.prepare_graft_phase3_multibug import prepare
from scripts.seed_graft_phase3_cache import seed


SOURCE = Path("configs/graft_independent_qwen7b_control.json")
BUGS = ("Time-1", "Lang-1", "Math-1")


def fixture(tmp_path):
    cases = []
    for bug in BUGS:
        checkout = tmp_path / "checkouts" / bug
        checkout.mkdir(parents=True)
        evidence = tmp_path / "evidence" / (bug + ".txt")
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("Synthetic fixture, not real failing test evidence")
        cases.append({
            "bug": bug,
            "checkout": str(checkout),
            "evidence_file": str(evidence),
            "evidence_kind": "observed_failure",
            "gold": {"files": ["src/main/java/Example.java"], "functions": []},
        })
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "cases": cases,
        "not_original_flexfl_condition": True,
    }))
    newconfig_path = prepare(SOURCE, dataset, tmp_path / "baseline")
    newconfig = json.loads(newconfig_path.read_text())
    assert newconfig["benchmark"]["budget"] == 90000

    old = deepcopy(newconfig)
    old["benchmark"]["budget"] = 40000
    old["benchmark"]["cache_dir"] = None
    source_run = tmp_path / "archived-run"
    (source_run / "cache").mkdir(parents=True)
    (source_run / "manifest.json").write_text(json.dumps({
        "config": old,
        "dataset_source": {
            "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        },
    }))
    (source_run / "accounting.json").write_text(json.dumps({
        "synthetic": False, "usage_unknown": False, "pending": False,
        "new_calls": 1, "physical_inference_tokens": 123,
    }))
    model_digest = "a" * 64
    (source_run / "environment.json").write_text(json.dumps({
        "provider": {"digest": model_digest},
    }))
    response = {
        "done": True,
        "prompt_eval_count": 110,
        "eval_count": 13,
        "message": {"content": '{"results":[]}'},
    }
    payload = {
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "synthetic fixture"}],
        "options": {
            "num_ctx": newconfig["benchmark"]["context_window"],
            "num_predict": newconfig["benchmark"]["output_tokens"],
            "seed": newconfig["benchmark"]["seed"],
            "temperature": newconfig["benchmark"]["temperature"],
        },
    }
    key = "b" * 64
    cache = {
        "identity": {
            "fingerprint": key,
            "request": payload,
            "model_digest": model_digest,
        },
        "response": response,
        "response_sha256": fingerprint(response),
    }
    (source_run / "cache" / (key + ".json")).write_text(json.dumps(cache))
    return dataset, source_run, newconfig


def test_seeds_verified_response_without_mutating_original(tmp_path):
    dataset, source_run, config = fixture(tmp_path)
    target = tmp_path / "fresh-cache"
    old_cache = next((source_run / "cache").glob("*.json"))
    old_digest = sha256(old_cache.read_bytes()).hexdigest()
    audit = seed(source_run, target, dataset=dataset, config=config)
    assert audit["seeded_response_files"] == 1
    assert audit["seeded_provider_tokens_historical_not_new"] == 123
    assert audit["new_provider_calls_during_seeding"] == 0
    assert not (target / "journal.sqlite3").exists()
    assert sha256(old_cache.read_bytes()).hexdigest() == old_digest
    assert sha256(next(target.glob("*.json")).read_bytes()).hexdigest() == old_digest


def test_seed_rejects_corruption_before_copy(tmp_path):
    dataset, source_run, config = fixture(tmp_path)
    cached = next((source_run / "cache").glob("*.json"))
    item = json.loads(cached.read_text())
    item["response_sha256"] = "0" * 64
    cached.write_text(json.dumps(item))
    target = tmp_path / "rejected-cache"
    with pytest.raises(CompatibilityError, match="Unverified Qwen cache response"):
        seed(source_run, target, dataset=dataset, config=config)
    assert not target.exists()


def test_seed_rejects_changed_cohort_or_prompt_settings(tmp_path):
    dataset, source_run, config = fixture(tmp_path)
    changed = deepcopy(config)
    changed["benchmark"]["context_window"] += 1
    with pytest.raises(CompatibilityError, match="request input"):
        seed(source_run, tmp_path / "bad-settings", dataset=dataset, config=changed)
    changed = deepcopy(config)
    changed["benchmark"]["candidate_count"] = 20
    with pytest.raises(CompatibilityError, match="request input"):
        seed(source_run, tmp_path / "bad-count", dataset=dataset, config=changed)


def test_preparation_seeds_isolated_cache_and_keeps_original_counters(tmp_path):
    dataset, source_run, _ = fixture(tmp_path)
    target = tmp_path / "new-policy-budget"
    config_path = prepare(SOURCE, dataset, target, seed_run=source_run)
    conf = json.loads(config_path.read_text())
    b = conf["benchmark"]
    assert b["budget"] == 90000
    assert b["physical_budget"] == 240000
    assert b["per_bug_physical_budget"] == 80000
    assert b["max_new_calls"] == 135
    assert b["cache_dir"] == str((target / "cache").resolve())
    assert len(list((target / "cache").glob("*.json"))) == 1
    audit = json.loads((target / "cache_seed_provenance.json").read_text())
    assert audit["new_provider_calls_during_seeding"] == 0
    assert audit["original_sqlite_journal_copied"] is False



def test_seed_reuses_followup_run_sibling_cache(tmp_path):
    """The completed 90k policy run stores responses beside run/, not inside."""
    dataset, source_run, config = fixture(tmp_path)
    manifest_file = source_run / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    sibling = source_run.parent / "cache"
    sibling.mkdir()
    previous = next((source_run / "cache").glob("*.json"))
    previous.rename(sibling / previous.name)
    manifest["config"]["benchmark"]["cache_dir"] = str(sibling.resolve())
    manifest_file.write_text(json.dumps(manifest))
    audit = seed(source_run, tmp_path / "third-run-cache", dataset=dataset, config=config)
    assert audit["source_verified_cache_dir"] == str(sibling.resolve())
    assert audit["seeded_response_files"] == 1
    assert audit["new_provider_calls_during_seeding"] == 0


def test_seed_rejects_external_cache_location(tmp_path):
    dataset, source_run, config = fixture(tmp_path)
    manifest_file = source_run / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    external = tmp_path / "elsewhere" / "cache"
    external.mkdir(parents=True)
    manifest["config"]["benchmark"]["cache_dir"] = str(external.resolve())
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(CompatibilityError, match="Source cache must be under"):
        seed(source_run, tmp_path / "unsafe-cache", dataset=dataset, config=config)
