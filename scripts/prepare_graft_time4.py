#!/usr/bin/env python3
"""Prepare one pre-screened Time-4 bug, without sending inference calls.

Gold labels were used to choose Time-4 from the frozen twelve-bug exploratory
screen. They are never included in the online manifest or model requests.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess


BUG = "Time-4"
SHA = "bcb044669b4d1f8d334861ccbd169924d6ef3b54"
REPO = "https://github.com/JodaOrg/joda-time.git"
RANKERS = ("SBIR", "Ochiai", "BoostN")


def command(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stderr}")
    return result.stdout.strip()


def main():
    root = Path("results/graft_time4_qwen7b").resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkout = Path("data/rave_fl_checkouts/Time-4").resolve()
    flexfl = Path("references/FlexFL_Time25").resolve()
    if not (flexfl / ".git").is_dir():
        raise RuntimeError(f"Missing FlexFL checkout: {flexfl}")
    paths = [f"FlexFL/data/FL_results/{m}/Defects4J/{BUG}_method-susps.csv" for m in RANKERS]
    paths += [f"FlexFL/data/input/trigger_tests/Defects4J/{BUG}.txt"]
    subprocess.run(["git", "-C", str(flexfl), "sparse-checkout", "add", *paths], check=True)
    for p in paths:
        if not (flexfl / p).is_file() or not (flexfl / p).stat().st_size:
            raise RuntimeError(f"Missing input: {flexfl / p}")
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        temp = checkout.with_name(checkout.name + ".preparing")
        if temp.exists():
            raise RuntimeError(f"Inspect interrupted checkout preparation first: {temp}")
        command("git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(temp))
        command("git", "-C", str(temp), "checkout", "--detach", SHA)
        if command("git", "-C", str(temp), "rev-parse", "HEAD") != SHA:
            raise RuntimeError("Pinned revision mismatch")
        temp.rename(checkout)
    if command("git", "-C", str(checkout), "rev-parse", "HEAD") != SHA:
        raise RuntimeError("Existing checkout has wrong revision; will not overwrite")
    if command("git", "-C", str(checkout), "status", "--porcelain"):
        raise RuntimeError("Buggy checkout must be clean")

    trigger = flexfl / paths[-1]
    observed = checkout / "failing_tests"
    if observed.is_file() and observed.read_text(errors="replace").strip():
        source = observed
        kind = "observed_failure"
        evidence = re.sub(r"(?m)^---\s+(.+)$", r"TEST: \1", source.read_text(errors="replace"))
    else:
        source = trigger
        kind = "trigger_description"
        evidence = trigger.read_text(errors="replace")
    evidence_file = root / "evidence" / "Time-4.txt"
    evidence_file.parent.mkdir(exist_ok=True)
    evidence_file.write_text(evidence)
    manifest = {
        "path_base": "cwd",
        "instances": [{
            "bug": BUG,
            "checkout": str(checkout),
            "flexfl_repo": str(flexfl),
            "buggy_revision": SHA,
            "source_repository": REPO,
            "evidence_file": str(evidence_file),
            "evidence_kind": kind
        }],
        "selection_rule": "Time-4 only, offline rank-10 bug among 12 frozen exploratory bugs",
        "evaluation_role": "exploratory_gold_stratified_not_held_out"
    }
    manifest_file = root / "instances.json"
    manifest_file.write_text(json.dumps(manifest, indent=2)+"\n")
    (root / "evidence_provenance.json").write_text(json.dumps({
        "kind":kind,"source":str(source),
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "prepared_sha256":hashlib.sha256(evidence_file.read_bytes()).hexdigest(),
    }, indent=2)+"\n")
    for arm in ("control", "anchored"):
        cfg = json.loads(Path(f"configs/graft_independent_qwen7b_{arm}.json").read_text())
        cfg["benchmark"]["dataset_file"] = str(manifest_file)
        cfg["benchmark"]["bugs"] = [BUG]
        cfg["benchmark"]["run_id"] = f"graft_time4_qwen7b_{arm}"
        # Both arms share candidate selection, backend and token/call budgets.
        (root / f"config_{arm}.json").write_text(json.dumps(cfg, indent=2)+"\n")
    print(f"Prepared Time-4 from pinned buggy revision {SHA}")
    print(f"Evidence: {kind} ({source})")
    print(f"Configs and results: {root}")


if __name__ == "__main__":
    main()
