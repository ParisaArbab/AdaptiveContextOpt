"""Frozen, project-balanced 100-bug Defects4J cohort and real-test collection.

No model calls. All bug IDs are chosen with a deterministic seed from the
active official Defects4J IDs *before* checking failures or reading gold.
Never substitute unavailable bugs. Existing 3-bug experiment is unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import shutil
import subprocess

from scripts.prepare_graft_phase3_real_bugs import collect_case

PROJECTS = ("Chart", "Closure", "Lang", "Math", "Time")
PER_PROJECT = 20
BATCH_COUNT = 20
BATCH_SIZE = 5
DEFAULT_SEED = 42
DEVELOPMENT_BATCHES = 4
PREVIOUS_DEVELOPMENT = ("Lang-1", "Math-1", "Time-1", "Time-4")
SCHEMA = "graft-100-frozen-stratified-defects4j-v1"


def digest(b: bytes) -> str:
    return sha256(b).hexdigest()


def write_new(path: Path, value: dict):
    if path.exists():
        raise FileExistsError("Refusing overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_cohort(cohort: dict):
    if cohort.get("schema") != SCHEMA or len(cohort.get("bugs", [])) != 100:
        raise ValueError("Requires a frozen 100-ID cohort")
    rows = cohort["bugs"]
    if len({r["bug"] for r in rows}) != 100:
        raise ValueError("Duplicate frozen bug IDs")
    if set(r["project"] for r in rows) != set(PROJECTS):
        raise ValueError("Missing selected project")
    if Counter(r["project"] for r in rows) != dict.fromkeys(PROJECTS, 20):
        raise ValueError("Cohort is not exactly 20 per project")
    for index, row in enumerate(rows):
        if row["batch"] != index // BATCH_SIZE:
            raise ValueError("Frozen batch position changed")
        if row["split"] != ("development" if row["batch"] < DEVELOPMENT_BATCHES else "evaluation"):
            raise ValueError("Frozen development/evaluation split changed")
        if row["project"] != PROJECTS[index % BATCH_SIZE]:
            raise ValueError("Frozen project rotation changed")
        if row["bug"] in PREVIOUS_DEVELOPMENT:
            raise ValueError("Previously analyzed bug in held-out cohort")
        if not re.fullmatch(re.escape(row["project"]) + r"-[1-9]\d*", row["bug"]):
            raise ValueError("Malformed frozen official bug ID")
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    if cohort["selection_sha256"] != digest(canonical):
        raise ValueError("Frozen 100-ID selection has changed")


def official_active_ids(defects4j: str, project: str) -> list[int]:
    proc = subprocess.run(
        [defects4j, "bids", "-p", project],
        text=True, capture_output=True, timeout=120, check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"Official bids failed for {project}: {proc.stderr[-350:]}")
    # Defects4J bids is a whitespace-separated list of integer IDs. A CLI
    # that returns headings or warnings instead must be investigated, not read
    # as though it contained genuine bug IDs.
    tokens = proc.stdout.replace(",", " ").split()
    if not tokens or any(not re.fullmatch(r"[1-9]\d*", t) for t in tokens):
        raise ValueError(f"Unexpected official active-bug output for {project}: {proc.stdout[:280]!r}")
    ids = sorted(set(int(t) for t in tokens))
    if len(ids) != len(tokens):
        raise ValueError(f"Duplicate official active-bug IDs: {project}")
    return ids


def freeze(defects4j: str, root: Path, seed: int = DEFAULT_SEED):
    if root.exists():
        raise FileExistsError(f"Frozen cohort directory already exists: {root}")
    binary = shutil.which(defects4j) or (str(Path(defects4j).resolve()) if Path(defects4j).is_file() else None)
    if binary is None:
        raise RuntimeError("Official defects4j binary missing")
    binary = str(Path(binary).resolve())
    source_repo = Path(binary).parents[2]
    revision = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Defects4J framework revision is not a Git SHA")
    by_project = {}
    active = {}
    for project in PROJECTS:
        ids = official_active_ids(binary, project)
        active[project] = ids
        excluded = {int(x.split("-")[1]) for x in PREVIOUS_DEVELOPMENT if x.startswith(project + "-")}
        eligible = [n for n in ids if n not in excluded]
        if len(eligible) < PER_PROJECT:
            raise ValueError(f"Not enough active, unused bugs in {project}")
        rng = random.Random(f"{SCHEMA}:{seed}:{project}")
        by_project[project] = rng.sample(eligible, PER_PROJECT)

    rows = []
    for batch in range(BATCH_COUNT):
        for project in PROJECTS:
            number = by_project[project][batch]
            rows.append({
                "bug": f"{project}-{number}",
                "project": project,
                "number": number,
                "batch": batch,
                "split": "development" if batch < DEVELOPMENT_BATCHES else "evaluation",
            })
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    cohort = {
        "schema": SCHEMA,
        "seed": seed,
        "quota_each_project": PER_PROJECT,
        "projects": list(PROJECTS),
        "excluded_previously_analyzed": list(PREVIOUS_DEVELOPMENT),
        "defects4j_framework_revision": revision,
        "defects4j_executable": binary,
        "official_active_bug_ids": active,
        "selection_sha256": digest(canonical),
        "source": "defects4j bids -p <project>, local installed release",
        "selection_uses_results_or_gold": False,
        "bugs": rows,
        "total_predeclared_bugs": 100,
        "development_count": 20,
        "evaluation_count": 80,
        "batch_count": BATCH_COUNT,
        "batch_size": BATCH_SIZE,
        "new_model_calls": 0,
    }
    validate_cohort(cohort)
    write_new(root / "cohort.json", cohort)
    write_new(root / "splits.json", {
        "development": [r["bug"] for r in rows if r["split"] == "development"],
        "evaluation": [r["bug"] for r in rows if r["split"] == "evaluation"],
    })
    print("FROZEN: 100 official active bug IDs, 20 per project, 20 fixed batches")
    print("Selection SHA256:", cohort["selection_sha256"])
    print("First batch:", [r["bug"] for r in rows[:5]])
    print("Cohort manifest:", root / "cohort.json")
    return cohort


def expected_batch(cohort: dict, batch: int) -> list[dict]:
    validate_cohort(cohort)
    if not 0 <= batch < BATCH_COUNT:
        raise ValueError("Batch must be between 0 and 19")
    rows = [r for r in cohort["bugs"] if r["batch"] == batch]
    if len(rows) != BATCH_SIZE:
        raise ValueError("Batch has missing predeclared IDs")
    return rows


def collect_batch(root: Path, cohort: dict, batch: int, cli: str, max_bytes: int = 9000):
    rows = expected_batch(cohort, batch)
    folder = (root / "batches" / f"batch_{batch:02d}").resolve()
    prepared = folder / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    ready = []
    audits = []
    issues = []
    for row in rows:
        bug = row["bug"]
        case_path = prepared / "records" / f"{bug}.json"
        audit_path = prepared / "audits" / f"{bug}.json"
        try:
            if case_path.exists() and audit_path.exists():
                case = json.loads(case_path.read_text())
                audit = json.loads(audit_path.read_text())
                if (
                    case.get("bug") != bug
                    or not Path(case["checkout"]).is_dir()
                    or not Path(case["evidence_file"]).is_file()
                    or digest(Path(case["evidence_file"]).read_bytes()) != audit["evidence_sha256"]
                ):
                    raise ValueError("Existing prepared record failed integrity check")
            elif case_path.exists() or audit_path.exists():
                raise RuntimeError("Partial archived record; manual audit required")
            else:
                case, audit = collect_case(
                    {"bug": bug, "source_repository": row["project"]},
                    cli, prepared, max_bytes=max_bytes,
                )
                write_new(case_path, case)
                write_new(audit_path, audit)
            ready.append(case)
            audits.append(audit)
            print(f"READY {bug}: {audit['test_method']}", flush=True)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            issues.append({"bug": bug, "error": str(exc)})
            print(f"UNAVAILABLE {bug}: {exc}", flush=True)
        # Save progress even if ssh disconnects; no synthetic replacements.
        report = {
            "schema": "graft-100-collection-report-v1",
            "batch": batch,
            "split": rows[0]["split"],
            "cohort_selection_sha256": cohort["selection_sha256"],
            "predeclared_bug_ids": [r["bug"] for r in rows],
            "prepared_bug_ids": [r["bug"] for r in ready],
            "errors": issues,
            "audits": audits,
            "status": "ready" if len(ready) == BATCH_SIZE else "incomplete",
            "model_calls": 0,
        }
        (prepared / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if len(ready) == BATCH_SIZE:
        dataset = {
            "schema": "graft-100-prepared-batch-v1",
            "not_original_flexfl_condition": True,
            "path_base": "cwd",
            "cohort_selection_sha256": cohort["selection_sha256"],
            "batch_index": batch,
            "cases": ready,
        }
        manifest = prepared / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != dataset:
                raise ValueError("Prepared manifest differs from immutable prior version")
        else:
            write_new(manifest, dataset)
    print("PREPARATION:", report["status"], len(ready), "/ 5")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    for action in ("freeze", "collect", "inspect"):
        sub = s.add_parser(action)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--defects4j", default="defects4j")
        if action == "freeze":
            sub.add_argument("--seed", type=int, default=DEFAULT_SEED)
        if action in ("collect", "inspect"):
            sub.add_argument("--batch", type=int, required=True)
        if action == "collect":
            sub.add_argument("--allow-real-tests", action="store_true")
            sub.add_argument("--max-evidence-bytes", type=int, default=9000)
    args = p.parse_args(argv)
    if args.command == "freeze":
        freeze(args.defects4j, args.root, args.seed)
        return 0
    cohort = json.loads((args.root / "cohort.json").read_text())
    expected = expected_batch(cohort, args.batch)
    if args.command == "inspect":
        print(json.dumps({"batch": args.batch, "bugs": expected, "new_model_calls": 0}, indent=2))
        return 0
    if not args.allow_real_tests:
        p.error("--allow-real-tests is required to check out and run real Defects4J tests")
    if args.max_evidence_bytes < 1000:
        p.error("At least 1000 complete evidence bytes required")
    result = collect_batch(
        args.root, cohort, args.batch, args.defects4j, args.max_evidence_bytes
    )
    return 0 if result["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
