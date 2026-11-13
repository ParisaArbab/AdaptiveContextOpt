"""Freeze a nested, project-balanced holdout using only pinned benchmark metadata."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

from rave_fl.eval.schema import fingerprint
from rave_fl.graft.prompt_settlement import require

ROOT = Path(__file__).resolve().parents[2]
D4J = "8c16da8230843cdc918eaf4ddb449637f02b83c6"
FLEX = "744143aa3e9c222b111beca55f5e75eab6d16285"
SEED = "graft-fl-linux-holdout-v1-20261113"
RANKERS = ("SBIR", "Ochiai", "BoostN")
EXCLUSIONS = ROOT / "configs/graft_server/exclusions-v1.json"
MANIFEST = ROOT / "configs/graft_server/selection-v1.json"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args],
                                   env={**os.environ, "GIT_ALLOW_PROTOCOL": "", "GIT_NO_LAZY_FETCH": "1",
                                        "GIT_OPTIONAL_LOCKS": "0"})


def order(kind, value):
    return hashlib.sha256(f"{SEED}|{kind}|{value}".encode()).hexdigest(), value


def generate(d4j, flex, exclusions):
    metadata = git(d4j, "ls-tree", "-r", "--name-only", D4J, "--", "framework/projects")
    revisions = {}
    for file in metadata.decode().splitlines():
        if not file.endswith("/active-bugs.csv"):
            continue
        project = Path(file).parent.name
        for row in csv.DictReader(io.StringIO(git(d4j, "show", f"{D4J}:{file}").decode())):
            revisions[f"{project}-{row['bug.id']}"] = (row["revision.id.buggy"], row["revision.id.fixed"])
    ranks = {}
    for ranker in RANKERS:
        prefix = f"FlexFL/data/FL_results/{ranker}/Defects4J"
        rows = git(flex, "ls-tree", "-r", FLEX, "--", prefix)
        ranks[ranker] = {Path(path).name.removesuffix("_method-susps.csv"): (spec.split()[2], path)
                         for spec, path in (line.split("\t") for line in rows.decode().splitlines())
                         if path.endswith("_method-susps.csv")}
    excluded = set().union(*exclusions.values())
    intersection = set(revisions).intersection(*ranks.values())
    eligible = {bug for bug in intersection - excluded
                if all(re.fullmatch("[0-9a-f]{40}", revision) for revision in revisions[bug])}
    projects = sorted({bug.rsplit("-", 1)[0] for bug in eligible}, key=lambda p: order("project", p))
    by_project = {p: iter(sorted((b for b in eligible if b.rsplit("-", 1)[0] == p),
                                 key=lambda b: order("defect", b))) for p in projects}
    ids = []
    while len(ids) < 200:
        previous = len(ids)
        for p in projects:
            bug = next(by_project[p], None)
            if bug:
                ids.append(bug)
            if len(ids) == 200:
                break
        require(len(ids) > previous, "Pinned sampling frame lacks 200 eligible defects")
    selected = [{"defect": bug, "buggy": revisions[bug][0], "fixed_offline_only": revisions[bug][1],
                 "rankings": {r: dict(zip(("git_blob", "path"), ranks[r][bug], strict=True)) for r in RANKERS}}
                for bug in ids]
    gt = "FlexFL/data/input/ground_truth/Defects4J/gt.json"
    label_blob = git(flex, "ls-tree", FLEX, "--", gt).decode().split()[2]
    manifest = {"schema": "graft-linux-holdout-v1", "seed": SEED, "defects4j": D4J, "flexfl": FLEX,
                "eligibility": "Active pinned Defects4J Git revisions with all three pinned FlexFL method CSVs and no historical/pilot overlap. SVN excluded prospectively for acquisition homogeneity.",
                "ordering": "SHA256 seed|project|name and seed|defect|ID, lexical tie-break; one defect per project per round",
                "excluded_ids_sha256": fingerprint(exclusions), "eligible_count": len(eligible),
                "eligible_by_project": dict(Counter(sorted(b.rsplit('-', 1)[0] for b in eligible))),
                "label_reference_offline_only": {"path": gt, "git_blob": label_blob},
                "core": selected[:100], "extension": selected[100:]}
    return {**manifest, "sha256": fingerprint(manifest)}


def read_selection(path=MANIFEST):
    value = json.loads(path.read_text())
    body = {k: v for k, v in value.items() if k != "sha256"}
    exclusions = json.loads(EXCLUSIONS.read_text())
    rows = value["core"] + value["extension"]
    require(value["sha256"] == fingerprint(body) and value["schema"] == "graft-linux-holdout-v1"
            and value["defects4j"] == D4J and value["flexfl"] == FLEX
            and value["excluded_ids_sha256"] == fingerprint(exclusions)
            and len(value["core"]) == len(value["extension"]) == 100 and len({r["defect"] for r in rows}) == 200
            and not ({r["defect"] for r in rows} & set().union(*exclusions.values())), "Invalid holdout selection")
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--defects4j", type=Path, required=True)
    p.add_argument("--flexfl", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(generate(args.defects4j, args.flexfl, json.loads(EXCLUSIONS.read_text())), indent=2))
