"""Phase-3 bounded configuration, no inference or provider required."""
import json
from pathlib import Path

import pytest

from scripts.prepare_graft_test_aware_live import prepare


def fixture_config(tmp_path):
    dataset = tmp_path / "instances.json"
    dataset.write_text('{"cases":[{"bug":"Time-4"}]}')
    value = {
        "benchmark": {
            "dataset": "defects4j",
            "dataset_file": str(dataset),
            "bugs": ["Time-4"],
            "candidate_provider": "flexfl",
            "model": "qwen2.5:7b",
            "batch_size": 1,
            "granularity": "method",
            "representation": "l1",
            "workers": 1,
            "backend": "ollama",
            "admission": "strict",
            "evaluation_mode": "policy",
            "max_full": 1,
        },
        "graft": {
            "selector": "bm25",
            "input_budget": 4096,
            "failure_name_anchor": True,
            "guard_unsupported_graph_downgrade": True,
            "max_graph": 2,
            "max_full": 1,
        },
        "graft_policies": ["l1_graph_focused"],
    }
    file = tmp_path / "original.json"
    file.write_text(json.dumps(value))
    return file


def test_phase3_has_isolated_stop_control_and_budgets(tmp_path):
    source = fixture_config(tmp_path)
    out = prepare(source, tmp_path / "new")
    config = json.loads(out.read_text())
    assert config["benchmark"]["bugs"] == ["Time-4"]
    assert config["benchmark"]["policies"] == ["fixed_l1"]
    assert config["benchmark"]["fusions"] == ["rrf", "suspicion_rank_blend"]
    assert config["benchmark"]["max_new_calls"] == 45
    assert config["benchmark"]["physical_budget"] == 60000
    assert config["graft"]["input_budget"] == 4096
    assert config["graft_policies"] == [
        "l1_anchor", "l1_graph_focused",
        "l1_graph_test_aware_nostop", "l1_graph_test_aware",
    ]
    provenance = json.loads((out.parent / "provenance.json").read_text())
    assert provenance["gold_in_online_decisions"] is False
    assert provenance["new_calls_during_preparation"] == 0
    assert "only early stop" in provenance["stopping_ablation"]
    with pytest.raises(FileExistsError):
        prepare(source, out.parent)


def test_phase3_missing_dataset_and_changed_inputs_fail_closed(tmp_path):
    source = fixture_config(tmp_path)
    cfg = json.loads(source.read_text())
    cfg["benchmark"]["dataset_file"] = str(tmp_path / "not-existing.json")
    source.write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError):
        prepare(source, tmp_path / "bad-data")
    assert not (tmp_path / "bad-data").exists()

    cfg["benchmark"]["dataset_file"] = str(tmp_path / "instances.json")
    cfg["benchmark"]["model"] = "other:7b"
    source.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Qwen"):
        prepare(source, tmp_path / "bad-model")

    cfg["benchmark"]["model"] = "qwen2.5:7b"
    cfg["graft"]["input_budget"] = 8192
    source.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="4096"):
        prepare(source, tmp_path / "bad-budget")


def test_zero_provider_call_audit_uses_only_initial_llm_state():
    from scripts.audit_graft_test_aware_offline import inspect_instance
    folder = Path(__file__).resolve().parents[1] / (
        "results/graft_phase2_time4_live_20261113_202949/run/"
        "instances/Time-4_0d6a675374"
    )
    row = inspect_instance(folder, 4096)
    assert row["new_provider_calls"] == 0
    assert row["gold_labels_read"] is False
    assert row["new_test_aware"]["status"] == "ready"
    assert row["new_test_aware"]["input_units"] <= 4096
    assert row["new_test_aware"]["diagnostics"]["test_signal"]["full_test_verified"] is False
