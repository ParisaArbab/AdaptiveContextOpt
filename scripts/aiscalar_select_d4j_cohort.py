#!/usr/bin/env python3
"""Select a reproducible, cross-project Defects4J cohort with three real FlexFL ranks.

Selection uses only local Defects4J active bug metadata and git tree filenames.
It never reads ground truth or a model outcome, never runs tests or inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

FLEXFL_URL = "https://github.com/ParisaArbab/FlexFL_OriginalReplication.git"
RANKERS = ("SBIR", "Ochiai", "BoostN")
RX = re.compile(
    r"^FlexFL/data/FL_results/(SBIR|Ochiai|BoostN)/Defects4J/"
    r"([A-Za-z][A-Za-z0-9]*-[1-9][0-9]*)_method-susps\.csv$"
)


def call(command, timeout=300):
    r = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(map(str, command))}\n"
                           f"{r.stderr[-1200:]}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=100, help="Number of distinct bugs")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-per-project", type=int, default=15)
    ap.add_argument("--reference", default="references/FlexFL_Multi")
    ap.add_argument("--output", default="configs/aiscalar_d4j_100_selected.txt")
    ap.add_argument("--exclude", nargs="*", default=["Time-25"],
                    help="Exclude previously evaluated bugs before selecting")
    ap.add_argument("--summary", default="logs/d4j_cohort_selection.json")
    args = ap.parse_args()
    if args.count < 1 or args.max_per_project < 1:
        ap.error("count and max-per-project must be positive")
    ref = Path(args.reference).resolve()
    if not (ref / ".git").is_dir():
        if ref.exists():
            ap.error(f"Reference directory exists but is not a git repository: {ref}")
        ref.parent.mkdir(parents=True, exist_ok=True)
        print("Creating separate sparse FlexFL clone...", flush=True)
        call(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
              FLEXFL_URL, str(ref)], timeout=900)

    d4j = shutil.which("defects4j")
    if not d4j:
        fallback = Path.home() / "defects4j/framework/bin/defects4j"
        if fallback.is_file():
            d4j = str(fallback)
    if not d4j:
        ap.error("Defects4J executable not found")

    # Git tree listings cover committed CSV paths without fetching all CSV blobs.
    paths = call(["git", "-C", str(ref), "ls-tree", "-r", "--name-only",
                  "HEAD", "FlexFL/data/FL_results"]).splitlines()
    candidates = defaultdict(set)
    for path in paths:
        match = RX.fullmatch(path)
        if match:
            ranker, bug = match.groups()
            candidates[bug].add(ranker)
    triple = {bug for bug, methods in candidates.items() if set(RANKERS) <= methods}
    if not triple:
        ap.error("No bugs with all three FlexFL ranking files found")

    # Active IDs, not an arbitrary numeric sequence that might contain deprecated bugs.
    projects = sorted({bug.rsplit("-", 1)[0] for bug in triple})
    by_project = {}
    skipped_projects = {}
    for project in projects:
        try:
            output = call([d4j, "bids", "-p", project], timeout=60)
            ids = {int(line.strip()) for line in output.splitlines()
                   if re.fullmatch(r"[0-9]+", line.strip())}
            if not ids:
                skipped_projects[project] = "No active numeric bug IDs returned"
                continue
            selected = [bug for bug in triple if bug.startswith(project + "-")
                        and int(bug.rsplit("-", 1)[1]) in ids and bug not in args.exclude]
            random.Random(f"{args.seed}:{project}").shuffle(selected)
            by_project[project] = selected[:args.max_per_project]
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            skipped_projects[project] = str(exc)[-350:]

    # Round robin across projects to avoid preferentially sampling only Math or Closure.
    chosen = []
    project_names = sorted(by_project)
    for index in range(args.max_per_project):
        for project in project_names:
            if len(chosen) >= args.count:
                break
            if index < len(by_project[project]):
                chosen.append(by_project[project][index])
        if len(chosen) >= args.count:
            break
    if len(chosen) < args.count:
        raise SystemExit(
            f"Insufficient eligible bugs: requested {args.count}, selected {len(chosen)}; "
            "increase --max-per-project or inspect Defects4J project support."
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proposed = "\n".join(chosen) + "\n"
    if output_path.exists() and output_path.read_text() != proposed:
        raise SystemExit(f"Selection differs from existing {output_path}; choose a new --output")
    output_path.write_text(proposed)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "count": len(chosen), "seed": args.seed,
        "selection": "deterministic round robin across projects, active Defects4J IDs, all three FlexFL ranks, no gold",
        "excluded": args.exclude,
        "max_per_project": args.max_per_project,
        "available_triple_flexfl_rankings": len(triple),
        "selected_by_project": dict(Counter(b.rsplit("-", 1)[0] for b in chosen)),
        "skipped_projects": skipped_projects,
        "bugs": chosen,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Selected {len(chosen)} unique bugs across {len(summary['selected_by_project'])} projects")
    print("Project distribution:", summary["selected_by_project"])
    print("Bug list:", output_path)
    print("Selection metadata:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
