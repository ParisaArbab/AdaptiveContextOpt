"""Prepare real, provenance-labeled inputs for the independent GRAFT smoke.

Run explicitly on AIScalar. Never invent observed failures, revisions, labels,
candidate spans, or replace a preselected bug after inspecting outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

BUGS = ("Time-1", "Lang-1", "Math-1")
RANKERS = ("SBIR", "Ochiai", "BoostN")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, fetch: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path("configs/rave_independent_smoke.json").read_text())
    rows = base["cases"]
    assert tuple(r["bug"] for r in rows) == BUGS, "Preselected bug list changed"
    flexfl = Path("references/FlexFL_Time25").resolve()
    if not flexfl.is_dir():
        raise RuntimeError(f"FlexFL repository missing: {flexfl}")
    relative = [
        f"FlexFL/data/FL_results/{ranker}/Defects4J/{bug}_method-susps.csv"
        for bug in BUGS for ranker in RANKERS
    ]
    relative += [
        f"FlexFL/data/input/trigger_tests/Defects4J/{bug}.txt"
        for bug in BUGS
    ]
    if fetch and not all((flexfl / p).is_file() for p in relative):
        if not (flexfl / ".git").exists():
            raise RuntimeError("FlexFL reference has no Git checkout; cannot expand sparse paths")
        command = ["git", "-C", str(flexfl), "sparse-checkout", "add", *relative]
        print("Expanding explicit FlexFL sparse-checkout paths for selected bugs", flush=True)
        subprocess.run(command, check=True)

    failures = []
    result = []
    evidence_records = []
    for row in rows:
        bug = row["bug"]
        checkout = Path(row["checkout"]).resolve()
        if not checkout.is_dir():
            failures.append(f"{bug}: missing buggy checkout {checkout}")
        for ranker in RANKERS:
            file = flexfl / f"FlexFL/data/FL_results/{ranker}/Defects4J/{bug}_method-susps.csv"
            if not file.is_file() or not file.stat().st_size:
                failures.append(f"{bug}: missing {ranker} CSV: {file}")

        raw = checkout / "failing_tests"
        trigger = flexfl / f"FlexFL/data/input/trigger_tests/Defects4J/{bug}.txt"
        dest = (root / "evidence" / f"{bug}.txt").resolve()
        if raw.is_file() and raw.read_text(errors="replace").strip():
            text = raw.read_text(errors="replace")
            # Prefix real D4J test headers with TEST for the GRAFT anchor.
            # Preserve the original failure content, never rewrite assertion values.
            text = re.sub(r"(?m)^---\s+(.+)$", r"TEST: \1", text)
            kind = "observed_failure"
            source = raw
        elif trigger.is_file() and trigger.read_text(errors="replace").strip():
            text = trigger.read_text(errors="replace")
            kind = "trigger_description"
            source = trigger
        else:
            failures.append(f"{bug}: no nonempty failing_tests or FlexFL trigger description")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        result.append({
            "bug": bug,
            "checkout": str(checkout),
            "flexfl_repo": str(flexfl),
            "buggy_revision": row["buggy_revision"],
            "source_repository": row["source_repository"],
            "evidence_file": str(dest),
            "evidence_kind": kind,
        })
        evidence_records.append({
            "bug": bug, "kind": kind, "source": str(source),
            "source_sha256": sha256(source), "prepared_sha256": sha256(dest)
        })
    if failures:
        for f in failures:
            print("MISSING:", f, file=sys.stderr)
        raise RuntimeError("Input preparation stopped. No model calls were made.")
    manifest = (root / "instances.json").resolve()
    manifest.write_text(json.dumps({
        "path_base": "cwd",
        "instances": result,
        "selection_rule": base["selection_rule"],
        "evaluation_role": "exploratory_not_held_out",
    }, indent=2) + "\n")
    (root / "evidence_provenance.json").write_text(
        json.dumps(evidence_records, indent=2) + "\n"
    )
    for arm in ("control", "anchored"):
        cfg = json.loads(
            Path(f"configs/graft_independent_qwen7b_{arm}.json").read_text()
        )
        cfg["benchmark"]["dataset_file"] = str(manifest)
        cfg["benchmark"]["run_id"] = root.name + "_" + arm
        (root / f"config_{arm}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for e in evidence_records:
        print(f'{e["bug"]}: {e["kind"]}, evidence SHA-256 {e["prepared_sha256"]}')
    print("Prepared manifest:", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expand-flexfl", action="store_true",
        help="Explicitly fetch missing sparse FlexFL input paths via git"
    )
    args = parser.parse_args()
    prepare(args.output, args.expand_flexfl)
