"""Fresh, read-only GRAFT preflight for corrected 100-bug BM25 batches.

Uses archived configs and prepared manifests without touching archived
preflight.json, completion gates, model caches or Qwen. Runs the exact
GRAFT preflight, including graph construction, with the current source code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rave_fl.graft.runner import preflight, validate_config

BATCHES = (2, 6, 7, 11, 14, 19)
SELECTION_SHA = "a27224d014f0fa8c4c9ca431e3cd9e264342be1c58e40ce0024392df33c486b6"


def check(root: Path, out: Path, batches=BATCHES):
    cohort = json.loads((root / "aggregate" / "aggregate.json").read_text())
    if cohort["cohort_selection_sha256"] != SELECTION_SHA or cohort["n_predeclared_unique_bugs"] != 100:
        raise ValueError("Frozen cohort mismatch; refusing to inspect")
    rows = []
    for batch in batches:
        exp = root / "batches" / f"batch_{batch:02d}" / "experiment"
        original_path = exp / "config.json"
        original = json.loads(original_path.read_text())
        config = validate_config(original)
        # Validate content before preflight; do not modify any original field.
        assert config["benchmark"]["candidate_count"] == 24
        assert config["benchmark"]["candidate_provider"] == "bm25"
        assert len(config["benchmark"]["bugs"]) == 5
        assert config["benchmark"]["backend"] == "ollama"
        assert config["benchmark"]["dataset"] == "defects4j"
        result = preflight(config)
        statuses = result.get("candidate_preparations", [])
        good = (
            result.get("ready") is True
            and len(statuses) == 5
            and {r["instance_id"] for r in statuses} == set(config["benchmark"]["bugs"])
            and all(r.get("status") == "ready" and r.get("candidates") == 24 for r in statuses)
            and result.get("inference_performed") is False
            and result.get("external_tests_performed") is False
        )
        record = {
            "batch": batch,
            "ready": good,
            "preparation_error": result.get("preparation_error"),
            "provider_error": result.get("provider_error"),
            "candidates": statuses,
            "source_config": str(original_path),
            "selection_sha256": SELECTION_SHA,
        }
        rows.append(record)
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"batch_{batch:02d}_corrected_preflight.json"
        target.write_text(json.dumps({"summary": record, "full_preflight": result}, indent=2) + "\n")
        print(f"BATCH {batch:02d} {'READY' if good else 'NOT READY'}")
        for candidate in statuses:
            print(f"  {candidate['instance_id']}: {candidate['status']}, "
                  f"candidates={candidate.get('candidates', 'n/a')}, "
                  f"error={candidate.get('error', '')}")
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/graft_100_cohort_20261115"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/graft_100_recovery_audit/corrected_preflight"))
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)),
                        help="Comma-separated frozen batch IDs, defaults to six BM25-blocked batches")
    args = parser.parse_args()
    batches = tuple(int(v) for v in args.batches.split(","))
    if any(n not in BATCHES for n in batches) or len(set(batches)) != len(batches):
        raise ValueError("Only the six frozen BM25-failed batches are allowed")
    result = check(args.root, args.out, batches)
    print(f"Ready batches: {sum(r['ready'] for r in result)}/{len(result)}")
    print("No Qwen inference or Defects4J tests performed.")


if __name__ == "__main__":
    main()
