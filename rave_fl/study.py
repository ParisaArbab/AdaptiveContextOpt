"""Prepare leakage-aware train/calibration/test bug splits for a real study.

The existing pilot can process multiple cases using --max-bugs. This module
freezes project-level splits *before* any model training or policy tuning.
It never invents defects, logs, data, or experiments.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def study_split(manifest: dict, seed: int = 0, min_bugs: int = 9) -> dict:
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or len(cases) < min_bugs:
        raise ValueError(f"Need >= {min_bugs} real independent bug cases")
    seen, groups = set(), defaultdict(list)
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every manifest case must be an object")
        bug, project = case.get("bug"), case.get("project")
        if not isinstance(bug, str) or not bug or bug in seen:
            raise ValueError("Bug IDs must be unique nonempty strings")
        if not isinstance(project, str) or not project:
            raise ValueError(f"{bug}: explicit project ID is required for project holdout")
        if not case.get("checkout") or not case.get("flexfl_repo"):
            raise ValueError(f"{bug}: must supply actual checkout and reference repo paths")
        seen.add(bug)
        groups[project].append(bug)
    if len(groups) < 3:
        raise ValueError("At least three independent projects are required for held-out project splits")
    projects = sorted(groups)
    random.Random(seed).shuffle(projects)
    n = len(projects)
    train_end = max(1, int(0.6 * n))
    cal_end = min(n-1, max(train_end+1, int(0.8 * n)))
    split_projects = {
        "train": projects[:train_end],
        "calibration": projects[train_end:cal_end],
        "test": projects[cal_end:],
    }
    if any(not part for part in split_projects.values()):
        raise ValueError("Could not assign a nonempty project to each split")
    split = {key: sorted(bug for p in ps for bug in groups[p])
             for key, ps in split_projects.items()}
    return {"splits": split, "project_splits": split_projects,
            "seed": seed, "bug_count": len(cases), "project_count": len(groups),
            "protocol": "project-held-out; no repeated bug IDs",
            "warning": "This prepares evaluation only; risk training still requires true L0/L1/Full observations and both label classes."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Exact train_risk JSON split")
    p.add_argument("--metadata", type=Path, help="Optional project-level audit JSON")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-bugs", type=int, default=9)
    args = p.parse_args()
    result = study_split(json.loads(args.manifest.read_text()), args.seed, args.min_bugs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result["splits"], indent=2) + "\n")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
