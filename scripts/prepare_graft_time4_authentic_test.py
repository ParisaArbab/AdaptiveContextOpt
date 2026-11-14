"""Prepare an authentic full failing-test method for the Time-4 development bug.

Uses archived buggy checkout and trigger-description evidence; does not run
tests or a model. Unlike the new official multi-bug cohort, this condition is
still trigger-description evidence, NOT a newly observed failing execution.
No gold labels enter the online test evidence.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from scripts.prepare_graft_phase3_real_bugs import find_test_source
from rave_fl.graft.runner import validate_config


def prepare(input_file: Path, original_config: Path, output: Path,
            test_id: str = "org.joda.time.TestPartial_Basics::testWith3") -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source = json.loads(input_file.read_text())["online"]
    checkout = Path(source["checkout"])
    if not checkout.is_dir():
        raise FileNotFoundError(f"Missing immutable Time-4 checkout: {checkout}")
    if source.get("instance_id") != "Time-4" or source["evidence"]["kind"] != "trigger_description":
        raise ValueError("Expected saved Time-4 trigger-description source")
    proc = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True)
    revision = proc.stdout.strip()
    if revision != source["revision"]:
        raise ValueError("Buggy checkout does not match frozen Time-4 revision")
    dirty = subprocess.run(["git", "-C", str(checkout), "status",
                            "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise ValueError("Tracked buggy-revision source is dirty")
    test_file, full_method = find_test_source(checkout, test_id, "src/test/java")
    old_failure = source["evidence"]["text"]
    evidence = (
        "AUTHENTIC COMPLETE BUGGY-REVISION TEST METHOD:\n"
        f"TEST: {test_id}\n"
        f"TEST FILE: {test_file.relative_to(checkout)}\n"
        f"TEST FILE SHA256: {sha256(test_file.read_bytes()).hexdigest()}\n"
        + full_method.decode("utf-8")
        + "\nORIGINAL PROVIDED TRIGGER DESCRIPTION (NOT A NEW TEST RUN):\n"
        + old_failure
    )
    if len(evidence.encode()) > 3500:
        raise ValueError(
            "Full test plus original trigger exceed 3500 bytes; refusing "
            "to truncate or claim a graph package will fit"
        )
    config = json.loads(original_config.read_text())
    b = config["benchmark"]
    if b["bugs"] != ["Time-4"] or b["candidate_provider"] != "flexfl" or b["model"] != "qwen2.5:7b":
        raise ValueError("Existing Time-4 model and FlexFL inputs required")
    if config["graft"].get("input_budget") != 4096:
        raise ValueError("Same original graph input budget required")
    output.mkdir(parents=True)
    evidence_path = output / "full_test_plus_original_trigger.txt"
    evidence_path.write_text(evidence)
    manifest = {
        "schema": "graft-time4-authentic-test-source-v1",
        "cases": [{
            "bug": "Time-4",
            "repository": source["repository"],
            "buggy_revision": revision,
            "checkout": str(checkout.resolve()),
            "flexfl_repo": source["flexfl_repo"],
            "evidence_file": str(evidence_path.resolve()),
            "evidence_kind": "trigger_description",
            "source_root": ".",
        }],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    b["dataset_file"] = str(manifest_path.resolve())
    b["run_id"] = "graft_phase3_time4_authentic_test_" + output.name
    b["max_new_calls"] = 45
    b["physical_budget"] = 60000
    b["per_bug_physical_budget"] = 45000
    b["require_observed_failure"] = False
    config = validate_config(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "schema": "graft-time4-full-test-v1",
        "bug": "Time-4",
        "development_not_heldout": True,
        "new_test_execution_performed": False,
        "trigger_is_original_and_not_observed": True,
        "test_source_file": str(test_file),
        "test_source_sha256": sha256(test_file.read_bytes()).hexdigest(),
        "method_source_sha256": sha256(full_method).hexdigest(),
        "source_revision": revision,
        "original_trigger_sha256": sha256(old_failure.encode()).hexdigest(),
        "new_combined_evidence_sha256": sha256(evidence.encode()).hexdigest(),
        "new_provider_calls_during_preparation": 0,
        "original_input_json": str(input_file.resolve()),
    }, indent=2) + "\n")
    print("Prepared authentic full test for", test_id)
    print("Combined evidence bytes:", len(evidence.encode()))
    print("New dataset:", manifest_path)
    print("New config:", output / "config.json")
    return output / "config.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-file", type=Path, default=Path(
        "results/graft_phase3_test_aware_time4_20261113_210525/run/"
        "instances/Time-4_0d6a675374/input.json"))
    ap.add_argument("--source-config", type=Path, default=Path(
        "results/graft_phase3_test_aware_time4_20261113_210525/config.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--test-id", default="org.joda.time.TestPartial_Basics::testWith3")
    args = ap.parse_args()
    prepare(args.input_file, args.source_config, args.output, args.test_id)


if __name__ == "__main__":
    main()
