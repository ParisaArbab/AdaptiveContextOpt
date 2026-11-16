"""Create an isolated, auditable workspace for corrected-BM25 recovery.

Never modifies the frozen source cohort. Symlinks prepared evidence read-only
by convention, while all inference artifacts are written under a new root.
This setup makes no Qwen or Defects4J calls.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

BATCHES = (2, 6, 7, 11, 14, 19)
SELECTION_SHA = "a27224d014f0fa8c4c9ca431e3cd9e264342be1c58e40ce0024392df33c486b6"


def initialize(original: Path, target: Path):
    original = original.resolve()
    target = target.resolve()
    if original == target or original in target.parents or target in original.parents:
        raise ValueError("The recovery root must be separate from the frozen cohort root")
    cohort_file = original / "cohort.json"
    cohort = json.loads(cohort_file.read_text())
    if cohort.get("selection_sha256") != SELECTION_SHA:
        raise ValueError("Selection hash does not match frozen 100-bug cohort")
    from scripts.graft_100_dataset import validate_cohort, expected_batch
    validate_cohort(cohort)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Recovery workspace already exists; no setup overwrite allowed")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cohort_file, target / "cohort.json")
    manifest = {
        "schema": "graft-100-corrected-bm25-recovery-v1",
        "original_root": str(original),
        "recovery_root": str(target),
        "cohort_selection_sha256": SELECTION_SHA,
        "batch_ids": list(BATCHES),
        "protocol": "Same frozen 5 bugs per batch, original observed failures, 24 BM25 candidates",
        "implementation_change": "Deduplicate entity IDs after scoring, before top-24 limit",
        "original_evidence_is_symlinked": True,
        "source_cohort_unchanged": True,
        "new_inference_started_by_setup": False,
        "original_batch_status": "Never updated; original aggregate remains original",
    }
    for n in BATCHES:
        b = f"batch_{n:02d}"
        source = original / "batches" / b / "prepared"
        prep_report = json.loads((source / "preparation_report.json").read_text())
        expected = [r["bug"] for r in expected_batch(cohort, n)]
        if prep_report.get("status") != "ready" or prep_report.get("predeclared_bug_ids") != expected:
            raise ValueError(f"Batch {n} has incomplete or mismatched original evidence")
        prepared = target / "batches" / b / "prepared"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.symlink_to(source, target_is_directory=True)
    (target / "recovery_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, default=Path("data/graft_100_cohort_20261115"))
    p.add_argument("--target", type=Path, default=Path("results/graft_100_bm25_recovery_v1"))
    args = p.parse_args()
    manifest = initialize(args.original, args.target)
    print("Recovery workspace created:", manifest["recovery_root"])
    print("Original frozen evidence linked; original outputs untouched")
    print("Ready batches:", ",".join(str(n) for n in BATCHES))
    print("No model calls performed")
    print("For a single explicitly authorized Qwen recovery batch, run:")
    print("  python -m scripts.graft_100_run run --root "
          + manifest["recovery_root"] + " --batch 2 --allow-inference")


if __name__ == "__main__":
    main()
