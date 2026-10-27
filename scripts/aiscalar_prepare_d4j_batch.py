#!/usr/bin/env python3
"""Prepare REAL Defects4J bugs for independent RAVE-FL evaluation on AIScalar.

Checks out official buggy projects, runs Defects4J tests, retains raw failing_tests
and writes compact observed-only summaries and a local dataset manifest.

No gold labels, patches, or reference fixes are used to select bugs or evidence.
Run from the repository root. Does not invoke an LLM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SOURCE_URL = "https://github.com/ParisaArbab/FlexFL_OriginalReplication.git"
RANKERS = ("SBIR", "Ochiai", "BoostN")
D4J_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9]*)-([1-9][0-9]*)$")
INFRA_FRAMES = ("junit.", "org.junit.", "org.apache.tools.ant.",
                "java.", "jdk.", "sun.", "org.hamcrest.")


def command(args: list[str], log: Path, timeout: int = 900, cwd: Path | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        stream.write("COMMAND: " + " ".join(args) + "\n")
        stream.flush()
        try:
            result = subprocess.run(args, cwd=cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return result.returncode
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMED OUT after {timeout} seconds\n")
            return 124


def collect_failure_text(raw: str, limit: int = 3) -> tuple[str, int]:
    """Only copy observed test names, failure details and real application frames."""
    sections = re.split(r"(?m)^---\s+", raw)[1:]
    if not sections:
        raise ValueError("No Defects4J '--- test_name' failure sections")
    out = []
    for section in sections:
        lines = section.splitlines()
        if not lines or not lines[0].strip():
            raise ValueError("Empty Defects4J failing test name")
        name = lines[0].strip()
        nonframes = []
        app_frames = []
        for line in lines[1:]:
            value = line.strip()
            if not value:
                continue
            if value.startswith("at "):
                if not value[3:].startswith(INFRA_FRAMES):
                    app_frames.append(value)
            elif not value.startswith(("...", "Caused by: java.lang.reflect.")):
                nonframes.append(value)
        if not nonframes:
            raise ValueError(f"Missing observed error detail for {name}")
        # Copy at most four adjacent lines, preserving the observed assertion text.
        text = ("TEST: " + name + "\nFAILURE: " +
                "\n".join(nonframes[:4]) +
                "\nPROJECT FRAMES:\n" + "\n".join(app_frames[:limit]))
        out.append(text)
    return "\n\n".join(out), len(out)


def setup_reference(ref: Path, bugs: list[str], log: Path) -> None:
    if not (ref / ".git").exists():
        if ref.exists():
            raise RuntimeError(f"Reference path exists but is not a Git clone: {ref}")
        if command(["git", "clone", "--depth", "1", "--filter=blob:none",
                    "--sparse", SOURCE_URL, str(ref)], log, timeout=900):
            raise RuntimeError(f"FlexFL clone failed; see {log}")
    patterns = ["FlexFL/data/input/ground_truth/Defects4J/gt.json"]
    for bug in bugs:
        for method in RANKERS:
            patterns.append(f"FlexFL/data/FL_results/{method}/Defects4J/{bug}_method-susps.csv")
    if command(["git", "-C", str(ref), "sparse-checkout", "set", "--no-cone", *patterns],
               log.with_name("flexfl_sparse.log"), timeout=900):
        raise RuntimeError("FlexFL sparse checkout failed")
    if not (ref / patterns[0]).is_file():
        raise RuntimeError("Missing FlexFL offline ground truth file")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bugs", nargs="+", required=True,
                    help="Explicit IDs such as Time-24 Lang-1 Math-1; no cherry-picking by gold")
    ap.add_argument("--reference", default="references/FlexFL_Multi")
    ap.add_argument("--checkout-root", default="data/rave_fl_checkouts/multi")
    ap.add_argument("--evidence-root", default="data/rave_fl_observed_failures")
    ap.add_argument("--manifest", default="configs/aiscalar_multi_d4j.json")
    ap.add_argument("--test-timeout", type=int, default=900)
    ap.add_argument("--reuse-failing-tests", action="store_true",
                    help="Do not rerun tests if a previous failing_tests file exists")
    args = ap.parse_args()
    bugs = args.bugs
    if len(set(bugs)) != len(bugs) or any(not D4J_PATTERN.fullmatch(b) for b in bugs):
        ap.error("Provide unique Defects4J IDs such as Time-24, Lang-1")
    if args.test_timeout < 60:
        ap.error("--test-timeout must be at least 60 seconds")
    root = Path.cwd()
    if not (root / "rave_fl/eval/cli.py").is_file():
        ap.error("Run from the AdaptiveContextOpt_AdaptivePilot repository root")
    d4j = shutil.which("defects4j")
    if not d4j:
        fallback = Path.home() / "defects4j/framework/bin/defects4j"
        if fallback.is_file():
            d4j = str(fallback)
    if not d4j:
        ap.error("Defects4J executable missing; install/configure ~/defects4j first")
    ref = Path(args.reference).resolve()
    checkout_root = Path(args.checkout_root).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    logs = root / "logs/d4j_batch_preparation"
    logs.mkdir(parents=True, exist_ok=True)
    setup_reference(ref, bugs, logs / "flexfl_clone.log")
    rows = []
    statuses = {}
    for bug in bugs:
        project, number = D4J_PATTERN.fullmatch(bug).groups()
        checkout = checkout_root / bug
        status = {"bug": bug, "checkout": str(checkout)}
        statuses[bug] = status
        try:
            files = [ref / f"FlexFL/data/FL_results/{m}/Defects4J/{bug}_method-susps.csv"
                     for m in RANKERS]
            if any(not f.is_file() or f.stat().st_size == 0 for f in files):
                raise RuntimeError("One or more FlexFL SBIR/Ochiai/BoostN CSVs are missing")
            if checkout.exists() and not (checkout / ".defects4j.config").is_file():
                raise RuntimeError("Checkout path exists but is not a Defects4J checkout")
            if not checkout.exists():
                checkout.parent.mkdir(parents=True, exist_ok=True)
                rc = command([d4j, "checkout", "-p", project, "-v", f"{number}b",
                              "-w", str(checkout)], logs / f"{bug}_checkout.log", timeout=900)
                if rc:
                    raise RuntimeError(f"Defects4J checkout failed (exit {rc})")
            revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain",
                 "--untracked-files=no"], text=True
            ).strip()
            if dirty:
                raise RuntimeError("Buggy checkout has modified tracked files")
            failure = checkout / "failing_tests"
            if not (args.reuse_failing_tests and failure.is_file() and failure.stat().st_size):
                rc = command([d4j, "test", "-w", str(checkout)],
                             logs / f"{bug}_test.log", timeout=args.test_timeout,
                             cwd=checkout)
                status["test_exit_code"] = rc
                if rc == 124:
                    raise RuntimeError("Defects4J test timed out")
            if not failure.is_file() or not failure.stat().st_size:
                raise RuntimeError("Missing observed failing_tests after running Defects4J tests")
            evidence, n = collect_failure_text(failure.read_text(errors="replace"))
            evidence_root.mkdir(parents=True, exist_ok=True)
            summary = evidence_root / f"{bug}_failure_evidence.txt"
            if summary.exists() and summary.read_text() != evidence:
                raise RuntimeError(f"Existing evidence differs; preserve it: {summary}")
            summary.write_text(evidence)
            status.update(status="ready", revision=revision, failing_tests=n,
                          evidence_characters=len(evidence), evidence_path=str(summary))
            rows.append({
                "instance_id": bug,
                "repository": f"defects4j/{project}",
                "language": "java",
                "revision": revision,
                "checkout": str(checkout),
                "flexfl_repo": str(ref),
                "evidence_file": str(summary),
                "evidence_kind": "observed_failure",
                "issue": f"Observed failing Defects4J tests for {bug}",
            })
            print(f"READY {bug}: {n} observed failing tests, {len(evidence)} evidence chars",
                  flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            status.update(status="skipped", reason=str(exc))
            print(f"SKIPPED {bug}: {exc}", file=sys.stderr, flush=True)

    manifest = {"version": "defects4j-observed-multibug-v1",
                "path_base": "cwd", "instances": rows}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"Manifest differs from existing {manifest_path}; use a new --manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status_file = logs / "preparation_status.json"
    status_file.write_text(json.dumps(statuses, indent=2) + "\n")
    print(f"Prepared {len(rows)}/{len(bugs)} bugs; manifest: {manifest_path}")
    print(f"Status details: {status_file}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
