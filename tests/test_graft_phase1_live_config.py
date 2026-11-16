"""Tests for the bounded Phase-1 live configuration; no provider calls."""
from copy import deepcopy
import json

import pytest

from scripts.prepare_graft_phase1_live import prepare
from rave_fl.graft.runner import validate_config


def source_config(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"cases":[]}')
    conf = {
        "benchmark": {
            "dataset": "defects4j",
            "dataset_file": str(dataset),
            "bugs": ["Time-4"],
            "candidate_provider": "flexfl",
            "backend": "ollama",
            "model": "qwen2.5:7b",
            "batch_size": 1,
            "max_full": 1,
            "admission": "strict",
            "workers": 1,
            "evaluation_mode": "policy",
            "granularity": "method",
            "representation": "l1",
            "fusions": ["rrf"],
        },
        "graft": {
            "failure_name_anchor": True,
            "max_graph": 2,
            "max_full": 1,
        },
        "graft_policies": ["l1_graph", "graft_adaptive"],
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(conf))
    return source


def test_preparation_is_explicit_bounded_and_validated(tmp_path):
    source = source_config(tmp_path)
    path = prepare(source, tmp_path / "new")
    conf = validate_config(json.loads(path.read_text()))
    assert conf["benchmark"]["fusions"] == ["rrf", "suspicion_rank_blend"]
    assert conf["benchmark"]["max_new_calls"] == 45
    assert conf["benchmark"]["per_bug_physical_budget"] == 45000
    assert conf["graft_policies"] == [
        "l1_anchor", "l1_graph", "l1_graph_suspicion", "graft_adaptive"
    ]
    assert json.loads((path.parent / "provenance.json").read_text())["gold_used_for_online_decisions"] is False
    with pytest.raises(FileExistsError):
        prepare(source, path.parent)


def test_invalid_model_and_missing_dataset_fail_before_any_run(tmp_path):
    source = source_config(tmp_path)
    data = json.loads(source.read_text())
    data["benchmark"]["model"] = "other:7b"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Qwen"):
        prepare(source, tmp_path / "bad-model")
    assert not (tmp_path / "bad-model").exists()

    data["benchmark"]["model"] = "qwen2.5:7b"
    data["benchmark"]["dataset_file"] = str(tmp_path / "missing.json")
    source.write_text(json.dumps(data))
    with pytest.raises(FileNotFoundError):
        prepare(source, tmp_path / "no-data")
    assert not (tmp_path / "no-data").exists()
