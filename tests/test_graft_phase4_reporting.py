"""Exercise the fixed synthetic rehearsal and offline report against sealed artifacts."""
import csv
import json
import shutil

import pytest

from rave_fl.eval.schema import CodeEntity, CompatibilityError, GoldLabels, fingerprint
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.evaluation import evaluate_profile
from rave_fl.graft.fixed_evidence_reporting import SYNTHETIC_LABEL, audit_saved_run, report
from rave_fl.pilot import write_json
from scripts.rehearse_graft_fixed_evidence import rehearse
from test_graft_fixed_evidence_schedule import fixed_fixture


@pytest.fixture
def completed_rehearsal(fixed_fixture, tmp_path):
    value, _ = fixed_fixture
    output = tmp_path / "rehearsal"
    rehearse(value, value["labels_file"], output, physical_budget=60000)
    return output


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def hashes(directory):
    return {file.relative_to(directory): file.read_bytes() for file in directory.rglob("*") if file.is_file()}


def test_end_to_end_rehearsal_regenerates_without_runtime_resources(completed_rehearsal, tmp_path, monkeypatch):
    run = completed_rehearsal / "run"
    original = completed_rehearsal / "report"
    summary = json.loads((original / "summary.json").read_text())
    assert summary["synthetic"] and summary["selected_pairs"] == 6
    assert summary["pair_status_counts"] == {"complete": 5, "incomplete": 1}
    assert summary["cohorts"]["matched_complete_labeled"]["n"] == 3
    assert summary["cohorts"]["matched_full_candidate_recall"]["n"] == 2
    assert summary["spending"]["new_calls"] == 24
    assert summary["spending"]["physical_inference_tokens"] == 0
    assert sum(arm["logical_tokens"] for arm in summary["operational"].values()) == summary["spending"]["simulated_provider_tokens"]
    arms = {(r["instance_id"], r["arm"]): r for r in read_csv(original / "per_bug_arms.csv")}
    pairs = {r["instance_id"]: r for r in read_csv(original / "paired_differences.csv")}
    assert arms["missing-label", "T0"]["label_status"] == "missing_for_profile"
    assert arms["file-only-label", "T0"]["label_status"] == "method_labels_unavailable"
    assert arms["file-only-label", "T0"]["file_labels_available"] == "True"
    assert arms["candidate-miss", "T1"]["candidate_recall_status"] == "miss"
    assert pairs["candidate-miss"]["matched_complete_labeled"] == "True"
    assert pairs["candidate-miss"]["matched_full_candidate_recall"] == "False"
    assert float(arms["candidate-miss", "T0"]["MRR"]) == 0
    assert arms["invalid-verdict", "T0"]["invalid_calls"] == "1"
    assert pairs["invalid-verdict"]["delta_MRR"] == "" and pairs["invalid-verdict"]["delta_logical_tokens"] != ""
    assert len(read_csv(original / "evidence_selection.csv")) == 12
    assert all(row["synthetic"] == "True" for row in [*arms.values(), *pairs.values()])
    for name in ("labels.json", "instances.json", "embeddings"):
        (completed_rehearsal / name).rename(tmp_path / name)

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline regeneration touched live execution or labels")

    monkeypatch.setattr(fixed.GraftProvider, "execute", forbidden)
    monkeypatch.setattr(fixed, "prepare", forbidden)
    monkeypatch.setattr(fixed, "load_gold", forbidden)
    before = hashes(run)
    regenerated = tmp_path / "regenerated"
    assert report(run, regenerated) == summary
    assert hashes(original) == hashes(regenerated)
    assert hashes(run) == before
    figures = list((original / "figures").glob("*"))
    assert len(figures) == 10
    assert all(SYNTHETIC_LABEL in file.read_text() for file in figures if file.suffix == ".svg")


def test_infeasible_report_keeps_cost_and_accuracy_denominators_separate(completed_rehearsal):
    directory = completed_rehearsal / "denied-report"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["pair_status_counts"] == {"incomplete": 1}
    assert summary["spending"]["new_calls"] == 0
    cohort = summary["cohorts"]["matched_complete_labeled"]
    assert cohort["n"] == 0 and cohort["arms"]["T0"]["MRR"] is None
    rows = read_csv(directory / "per_bug_arms.csv")
    assert len(rows) == 2 and {r["arm_status"] for r in rows} == {"infeasible"}
    assert all("call limit" in r["plan_reasons"] for r in rows)
    assert "No joint-complete labeled pairs" in (directory / "figures/accuracy.svg").read_text()


@pytest.mark.parametrize("tamper", ["ranking", "usage", "request_order", "candidate", "non_synthetic"])
def test_offline_audit_rejects_inconsistent_saved_results(completed_rehearsal, tmp_path, tamper):
    source = tmp_path / "tampered"
    shutil.copytree(completed_rehearsal / "run", source)
    result = json.loads((source / "results.json").read_text())
    arm = result["pairs"][0]["arms"][fixed.T0]
    if tamper == "ranking":
        arm["ranked_functions"].reverse()
    elif tamper == "usage":
        arm["calls"][0]["input_tokens"] += 1
    elif tamper == "candidate":
        arm["calls"][0]["target"] = "out-of-schedule"
    elif tamper == "non_synthetic":
        result["synthetic"] = False
    else:
        ledger = source / "raw_ledgers/requests.jsonl"
        events = ledger.read_text().splitlines()
        first, second = [i for i, line in enumerate(events) if json.loads(line)["event"] == "request"][:2]
        events[first], events[second] = events[second], events[first]
        ledger.write_text("\n".join(events) + "\n")
    result.pop("sha256")
    write_json(source / "results.json", {**result, "sha256": fingerprint(result)})
    with pytest.raises(CompatibilityError, match={
        "ranking": "ranking reconstruction", "usage": "accounting mismatch", "request_order": "prefix",
        "candidate": "targets differ", "non_synthetic": "sealed fixed-evidence result",
    }[tamper]):
        audit_saved_run(source, None)


def test_pending_usage_cannot_be_reported_as_final_or_retried(fixed_fixture, tmp_path, monkeypatch):
    _, config = fixed_fixture
    attempts = 0

    def fail(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("synthetic provider interrupted")

    monkeypatch.setattr(fixed.GraftProvider, "execute", fail)
    run = tmp_path / "unknown"
    with pytest.raises(TimeoutError):
        fixed.run(config, run)
    with pytest.raises(CompatibilityError, match="Finalized results are missing"):
        report(run, tmp_path / "report")
    with pytest.raises(fixed.UnknownUsage):
        fixed.run(config, run, resume=True)
    assert attempts == 1
    assert json.loads((run / "accounting.json").read_text())["reserved_or_unknown_tokens"] == 4256


def test_strict_profile_preserves_out_of_pool_overload_as_zero():
    entity = CodeEntity("x.A.calculate(int)", "A.java", 1, 2, "int calculate(int n) { return n; }", 1.0)
    arm = {"ranked_functions": [entity.entity_id]}
    gold = GoldLabels(("A.java",), ("x.A.calculate(String)",))
    method = evaluate_profile(arm, [entity], gold, "defects4j", "java_strict_v2")["method"]
    assert method["candidate_pool_coverage"] == 0 and method["MRR"] == method["Top-5"] == 0
