"""Offline regression tests for frozen-cohort analysis; no Qwen needed."""
import json
from pathlib import Path

from scripts.graft_100_postmortem import diagnose, report
from scripts.graft_100_rf_feasibility import collect_development_labels, feasibility


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _fixture(tmp_path):
    root = tmp_path / "cohort"
    rows = []
    policies = ("l1_anchor", "l1_graph_focused",
                "l1_graph_test_aware_nostop", "l1_graph_test_aware")
    for batch in range(20):
        for i in range(5):
            bug = f"Fake{batch:02d}-{i}"
            for fusion in ("rrf", "suspicion_rank_blend"):
                rows.append({
                    "bug": bug, "batch": batch,
                    "split": "development" if batch < 4 else "evaluation",
                    "fusion": fusion, "status": "complete_with_graph",
                    "file_mrr": {p: 0.5 for p in policies},
                    "graph_calls": dict(zip(policies, (0, 2, 2, 2))),
                })
    _write(root / "aggregate" / "aggregate.json",
           {"cohort_selection_sha256": "frozen", "instances": rows})
    return root


def test_no_fake_gain_or_evaluation_leakage(tmp_path):
    root = _fixture(tmp_path)
    audit = diagnose(root)
    assert audit["fully_paired_unique_bugs"] == 100
    assert audit["fully_paired_evaluation_bugs"] == 80
    assert audit["incomplete_unique_bugs"] == 0
    labels = collect_development_labels(audit)
    assert len(labels) == 20
    assert {r["label"] for r in labels} == {0}
    rf = feasibility(root)
    assert rf["trained"] is False
    assert rf["learnable"] is False
    assert rf["evaluation_split_used_for_training"] is False
    assert rf["label_counts"] == {"0": 20}


def test_preflight_failure_remains_incomplete(tmp_path):
    root = _fixture(tmp_path)
    a = root / "aggregate" / "aggregate.json"
    j = json.loads(a.read_text())
    for row in j["instances"]:
        if row["bug"] == "Fake02-1":
            row["status"] = "prepared_only"
    _write(a, j)
    _write(root / "batches" / "batch_02" / "experiment" / "preflight.json", {
        "candidate_preparations": [
            {"instance_id": "Fake02-1", "status": "unavailable",
             "error": "No unique candidate pool"}
        ]
    })
    audit = diagnose(root)
    assert audit["fully_paired_unique_bugs"] == 99
    assert audit["incomplete_unique_bugs"] == 1
    assert audit["incomplete"][0]["reason"] == "candidate_pool"
    assert "Fake02-1" in report(audit)
