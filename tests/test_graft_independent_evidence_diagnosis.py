"""Per-bug FlexFL debugging-evidence diagnostics, no provider calls."""
import json
from pathlib import Path

from scripts.diagnose_graft_independent_evidence import inspect, main


def manifest(tmp_path):
    return {
        "path_base": "cwd",
        "cases": [
            {"bug": "Time-1", "flexfl_repo": str(tmp_path / "flexfl"), "checkout": str(tmp_path / "checkout")},
            {"bug": "Lang-1", "flexfl_repo": str(tmp_path / "flexfl"), "checkout": str(tmp_path / "checkout")},
        ],
    }


def test_detects_exact_missing_bug_without_fabricating_evidence(tmp_path):
    base = tmp_path / "flexfl" / "data" / "input" / "trigger_tests" / "Defects4J"
    base.mkdir(parents=True)
    (base / "Time-1.txt").write_text("testWith3() raised AssertionFailedError\n")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest(tmp_path)))
    report = inspect(path)
    assert report["bug_count"] == 2
    assert report["all_have_evidence"] is False
    assert report["new_qwen_calls"] == 0
    assert report["uses_gold_labels"] is False
    assert report["bugs"][0]["bug"] == "Time-1"
    assert report["bugs"][0]["evidence_available_to_current_adapter"]
    assert report["bugs"][1]["bug"] == "Lang-1"
    assert not report["bugs"][1]["evidence_available_to_current_adapter"]
    assert not report["bugs"][1]["sources"]["bug_reports"]["first_existing"]


def test_empty_file_is_not_counted_as_failure_evidence(tmp_path):
    base = tmp_path / "flexfl" / "data" / "input" / "trigger_tests" / "Defects4J"
    base.mkdir(parents=True)
    (base / "Time-1.txt").write_text("   \n")
    path = tmp_path / "manifest.json"
    data = manifest(tmp_path)
    data["cases"] = data["cases"][:1]
    path.write_text(json.dumps(data))
    report = inspect(path)
    assert not report["all_have_evidence"]
    assert report["bugs"][0]["sources"]["trigger_tests"]["first_existing"].endswith("Time-1.txt")


def test_explicit_real_evidence_is_valid_and_report_is_immutable(tmp_path):
    evidence = tmp_path / "real_failure.txt"
    evidence.write_text("Recorded assertion error from failing tests\n")
    path = tmp_path / "manifest.json"
    data = manifest(tmp_path)
    data["cases"] = data["cases"][:1]
    data["cases"][0]["evidence_file"] = str(evidence)
    data["cases"][0]["evidence_kind"] = "observed_failure"
    path.write_text(json.dumps(data))
    output = tmp_path / "audit.json"
    assert main(["--manifest", str(path), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["all_have_evidence"]
    assert report["bugs"][0]["explicit_evidence_file"]["nonempty"]
    import pytest
    with pytest.raises(FileExistsError):
        main(["--manifest", str(path), "--output", str(output)])
