"""Workspace creation must never overwrite the frozen experiment."""
import json
from pathlib import Path

import pytest

from scripts import graft_100_recovery_workspace as rec


def test_workspace_preserves_original(monkeypatch, tmp_path):
    original = tmp_path / "frozen"
    target = tmp_path / "recovery"
    cohort = {"selection_sha256": rec.SELECTION_SHA, "bugs": [{"bug": "A"}]}
    original.mkdir()
    (original / "cohort.json").write_text(json.dumps(cohort))
    monkeypatch.setattr("scripts.graft_100_dataset.validate_cohort", lambda x: None)
    monkeypatch.setattr("scripts.graft_100_dataset.expected_batch",
                        lambda cohort, batch: [{"bug": f"Bug-{batch}"}])
    for b in rec.BATCHES:
        p = original / "batches" / f"batch_{b:02d}" / "prepared"
        p.mkdir(parents=True)
        (p / "preparation_report.json").write_text(
            json.dumps({"status": "ready", "predeclared_bug_ids": [f"Bug-{b}"]})
        )
        (p / "manifest.json").write_text(json.dumps({"untouched": True}))
    before = (original / "batches" / "batch_02" / "prepared" / "manifest.json").read_bytes()
    result = rec.initialize(original, target)
    assert result["new_inference_started_by_setup"] is False
    assert (target / "batches" / "batch_02" / "prepared").is_symlink()
    assert (target / "batches" / "batch_02" / "prepared" / "manifest.json").read_bytes() == before
    assert (original / "batches" / "batch_02" / "prepared" / "manifest.json").read_bytes() == before
    assert not (target / "batches" / "batch_02" / "experiment").exists()
    with pytest.raises(FileExistsError):
        rec.initialize(original, target)


def test_rejects_target_below_original(tmp_path):
    root = tmp_path / "frozen"
    root.mkdir()
    with pytest.raises(ValueError):
        rec.initialize(root, root / "recovery")
