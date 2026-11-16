"""Input validation and frozen-baseline checks for the bounded Time-4 Phase-2 run."""
import json
import pytest
from scripts.prepare_graft_focused_time4_live import prepare


def source_config(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"cases":[{"bug":"Time-4"}]}')
    value = {
        "benchmark": {
            "dataset": "defects4j",
            "dataset_file": str(dataset),
            "bugs": ["Time-4"],
            "model": "qwen2.5:7b",
            "candidate_provider": "flexfl",
            "representation": "l1",
            "granularity": "method",
            "batch_size": 1,
            "backend": "ollama",
            "workers": 1,
            "admission": "strict",
            "evaluation_mode": "policy",
            "max_full": 1,
        },
        "graft": {
            "selector": "bm25",
            "input_budget": 4096,
            "failure_name_anchor": True,
            "max_graph": 2,
            "max_full": 1,
            "guard_unsupported_graph_downgrade": True,
        },
        "graft_policies": ["l1_graph_suspicion"],
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(value))
    return source


def test_prepare_preserves_fixed_controls_and_limits(tmp_path):
    source = source_config(tmp_path)
    output = tmp_path / "prepared"
    created = prepare(source, output)
    conf = json.loads(created.read_text())
    assert conf["benchmark"]["bugs"] == ["Time-4"]
    assert conf["benchmark"]["fusions"] == ["rrf", "suspicion_rank_blend"]
    assert conf["benchmark"]["max_new_calls"] == 40
    assert conf["benchmark"]["physical_budget"] == 55000
    assert conf["graft"]["input_budget"] == 4096
    assert conf["graft_policies"] == [
        "l1_anchor", "l1_graph_suspicion", "l1_graph_focused"
    ]
    prov = json.loads((output / "provenance.json").read_text())
    assert prov["gold_used_for_online_decisions"] is False
    assert prov["new_provider_calls_during_preparation"] == 0
    with pytest.raises(FileExistsError):
        prepare(source, output)


def test_fail_before_writing_when_dataset_missing(tmp_path):
    source = source_config(tmp_path)
    value = json.loads(source.read_text())
    value["benchmark"]["dataset_file"] = str(tmp_path / "missing.json")
    source.write_text(json.dumps(value))
    with pytest.raises(FileNotFoundError):
        prepare(source, tmp_path / "never")


def test_fail_on_wrong_model_and_changed_input_budget(tmp_path):
    source = source_config(tmp_path)
    conf = json.loads(source.read_text())
    conf["benchmark"]["model"] = "other:7b"
    source.write_text(json.dumps(conf))
    with pytest.raises(ValueError, match="Qwen"):
        prepare(source, tmp_path / "bad_model")
    conf["benchmark"]["model"] = "qwen2.5:7b"
    conf["graft"]["input_budget"] = 8000
    source.write_text(json.dumps(conf))
    with pytest.raises(ValueError, match="budget"):
        prepare(source, tmp_path / "bad_budget")
