from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pytest
from rave_fl.graft.cli import main
from rave_fl.graft.runner import validate_config, run, preflight
from rave_fl.graft.reporting import report
from rave_fl.eval.schema import CompatibilityError


def fixture_config(tmp_path):
    root = tmp_path / "fixture"
    assert main(["fixture", "--output", str(root)]) == 0
    return validate_config(json.loads((root / "graft.json").read_text()))


def test_end_to_end_two_languages_all_controls_and_immutable_resume(tmp_path):
    config = fixture_config(tmp_path)
    output = tmp_path / "run"
    first = run(config, output)
    assert set(first["statuses"].values()) == {"completed"}
    assert first["spending"]["physical_inference_tokens"] == 0
    summary = report(output)
    assert summary["synthetic"] and (output / "REPORT.md").exists()
    rows = json.loads((output / "results.json").read_text())
    assert len(rows) == 14
    assert {
        "prior",
        "full",
        "fixed_l1",
        "highest_suspicion",
        "v2_utility",
        "l1_graph",
        "graft_adaptive",
    } == {r["policy"] for r in rows}
    second = run(config, output, resume=True)
    assert first["spending"] == second["spending"]
    assert first["statuses"] == second["statuses"]
    changed = deepcopy(config)
    changed["graft"]["paths"] = True
    with pytest.raises(CompatibilityError, match="changed"):
        run(changed, output, resume=True)


def test_preflight_and_live_guard_do_not_send_requests(tmp_path):
    config = fixture_config(tmp_path)
    assert preflight(config)["ready"]
    config["benchmark"]["backend"] = "ollama"
    with pytest.raises(CompatibilityError, match="disabled"):
        run(config, tmp_path / "live")
    assert not (tmp_path / "live").exists()


def test_malformed_configs_and_fake_embeddings_are_not_silent_defaults(tmp_path):
    with pytest.raises(CompatibilityError, match="Unknown"):
        validate_config({"graftt": {}})
    with pytest.raises(CompatibilityError, match="strict"):
        validate_config({"benchmark": {"admission": "historical"}})
    with pytest.raises(CompatibilityError, match="embedding"):
        validate_config({"graft": {"selector": "topo"}})
    with pytest.raises(CompatibilityError, match="max_full"):
        validate_config({"graft": {"max_full": 2}})


def test_result_tampering_is_rejected_before_reusing_completed_arms(tmp_path):
    config = fixture_config(tmp_path)
    output = tmp_path / "run"
    first = run(config, output)
    statusfile = next((output / "instances").glob("*/status.json"))
    status = json.loads(statusfile.read_text())
    entry = next(iter(status["arms"].values()))
    (output / entry["file"]).write_text("{}")
    with pytest.raises(CompatibilityError, match="checksum"):
        run(config, output, resume=True)
    assert json.loads((output / "accounting.json").read_text()) == first["spending"]
    assert (output / entry["file"]).read_text() == "{}"


def test_incomplete_and_unlabeled_are_reported_separately(tmp_path):
    config = fixture_config(tmp_path)
    config["benchmark"]["budget"] = 0
    output = tmp_path / "limited"
    result = run(config, output)
    assert "incomplete" in result["statuses"].values()
    report(output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["arm_statuses"]["incomplete"] > 0
    assert not [r for r in summary["rows"] if r["cohort"] == "matched_completed_labeled"]
    operational = [
        r
        for r in summary["rows"]
        if r["cohort"] == "known_labeled_operational" and r["policy"] != "prior"
    ]
    assert operational and all(r["MRR"] == 0 for r in operational)


def test_failed_preparation_stays_in_operational_denominator(tmp_path):
    import shutil

    config = fixture_config(tmp_path)
    shutil.rmtree(tmp_path / "fixture/checkout")
    output = tmp_path / "missing-sources"
    result = run(config, output)
    assert set(result["statuses"].values()) == {"failed"}
    rows = json.loads((output / "results.json").read_text())
    assert len(rows) == 14 and all(r["status"] == "failed" for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["rows"] and all(r["n"] == 2 and r["MRR"] == 0 for r in summary["rows"])


def test_resource_identity_avoids_model_weights_and_source_root_remains_relative(tmp_path):
    from rave_fl.graft.runner import resource_identity

    config = validate_config({"benchmark": {"source_root": "src"}})
    assert config["benchmark"]["source_root"] == "src"
    tokenizer = tmp_path / "local-model"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}")
    (tokenizer / "chat_template.jinja").write_text("fixture template")
    (tokenizer / "weights.safetensors").write_text("fixture weights, deliberately excluded")
    config["tokenizer_dir"] = str(tokenizer)
    identity = resource_identity(config)
    assert len(identity) == 2 and not any(p.endswith(".safetensors") for p in identity)
