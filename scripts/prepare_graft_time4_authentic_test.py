"""Prepare an authentic official Time-4 trigger test with explicit provenance.

The triggering test was introduced after the raw buggy Git revision. Defects4J
records it in 4.test.patch. Only the test method is read from the pinned
fixed-revision test FILE; ALL production code stays at the immutable buggy
revision. This is trigger-description evidence, NOT a new observed test run.
No gold labels or fixed production source enter the model input.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from scripts.prepare_graft_phase3_real_bugs import _one_test_method
from rave_fl.graft.runner import validate_config

# These are the frozen, official Defects4J Time-4 revisions. The framework's
# 4.test.patch documents testWith3() as a benchmark triggering test that is
# absent from the raw buggy source but present in the fixed test revision.
# NEVER read the fixed revision's production source or change the buggy tree.
TIME4_BUGGY_REVISION = "bcb044669b4d1f8d334861ccbd169924d6ef3b54"
TIME4_FIXED_TEST_REVISION = "3ba9ba799b3261b7332a467a88be142c83b298fd"
OFFICIAL_TEST_PATH = "src/test/java/org/joda/time/TestPartial_Basics.java"
OFFICIAL_TEST_ID = "org.joda.time.TestPartial_Basics::testWith3"
OFFICIAL_DEFECTS4J_PATCH = (
    "https://github.com/rjust/defects4j/blob/master/"
    "framework/projects/Time/patches/4.test.patch"
)


def _verified_official_time4_test(checkout: Path, revision: str, test_id: str):
    """Get exactly the official trigger method, never fixed production code.

    No checkout, network, compilation or model calls. If the fixed test commit
    object is unavailable locally, fail closed rather than using a later
    release, a guessed snippet or the truncated trigger description.
    """
    if revision != TIME4_BUGGY_REVISION or test_id != OFFICIAL_TEST_ID:
        raise ValueError("Only the pinned official Time-4 trigger and buggy revision are supported")
    source_path = checkout / OFFICIAL_TEST_PATH
    if not source_path.is_file() or source_path.is_symlink():
        raise FileNotFoundError(f"Buggy checkout test file missing: {source_path}")
    raw_buggy = source_path.read_bytes()
    if _one_test_method(raw_buggy, "testWith3") is not None:
        raise ValueError(
            "Unexpected testWith3() in raw Time-4 buggy revision. "
            "Verify the checkout is not modified or benchmark-patched."
        )
    reference = TIME4_FIXED_TEST_REVISION + ":" + OFFICIAL_TEST_PATH
    proc = subprocess.run(
        ["git", "-C", str(checkout), "show", reference],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, timeout=60,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(
            "Official Time-4 fixed-revision TEST source is not in the local Git "
            "object store. Do not use a different revision. Obtain the pinned "
            f"commit {TIME4_FIXED_TEST_REVISION} or use the official "
            f"Defects4J test-patch workflow. Git error: {proc.stderr.decode(errors='replace')[-400:]}"
        )
    full_test = _one_test_method(proc.stdout, "testWith3")
    if full_test is None:
        raise ValueError("Official fixed-revision test file has no complete testWith3 method")
    if not all(pattern in full_test for pattern in (
        b"test.with(DateTimeFieldType.clockhourOfDay(), 6)",
        b"fail()",
        b"catch (IllegalArgumentException",
    )):
        raise ValueError("Pinned test method fails the expected Defects4J trigger signature")
    return full_test, {
        "test_source_revision": TIME4_FIXED_TEST_REVISION,
        "test_source_role": "official_defects4j_trigger_test_only",
        "test_source_path": OFFICIAL_TEST_PATH,
        "test_source_full_sha256": sha256(proc.stdout).hexdigest(),
        "test_method_sha256": sha256(full_test).hexdigest(),
        "buggy_test_file_sha256": sha256(raw_buggy).hexdigest(),
        "raw_buggy_revision_contains_test_method": False,
        "official_patch_reference": OFFICIAL_DEFECTS4J_PATCH,
        "fixed_production_source_read": False,
        "buggy_checkout_modified": False,
        "new_test_execution_performed": False,
    }



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
    full_method, test_provenance = _verified_official_time4_test(
        checkout, revision, test_id
    )
    old_failure = source["evidence"]["text"]
    evidence = (
        "AUTHENTIC OFFICIAL DEFECTS4J TRIGGER TEST METHOD:\n"
        "TEST METHOD SOURCE: FIXED-REVISION TEST FILE ONLY, NOT FIXED PRODUCTION CODE\n"
        f"TEST: {test_id}\n"
        f"TEST REVISION: {test_provenance['test_source_revision']}\n"
        f"TEST FILE: {test_provenance['test_source_path']}\n"
        f"TEST SOURCE SHA256: {test_provenance['test_source_full_sha256']}\n"
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
        "test_source_file": test_provenance["test_source_path"],
        "test_source_revision": test_provenance["test_source_revision"],
        "test_source_sha256": test_provenance["test_source_full_sha256"],
        "method_source_sha256": test_provenance["test_method_sha256"],
        "official_trigger_provenance": test_provenance,
        "source_revision": revision,
        "original_trigger_sha256": sha256(old_failure.encode()).hexdigest(),
        "new_combined_evidence_sha256": sha256(evidence.encode()).hexdigest(),
        "new_provider_calls_during_preparation": 0,
        "original_input_json": str(input_file.resolve()),
    }, indent=2) + "\n")
    print("Prepared official benchmark trigger test for", test_id)
    print("Trigger source revision:", test_provenance["test_source_revision"])
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
