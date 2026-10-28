"""Freeze deterministic, disjoint repository/project splits before inference."""
import argparse
import json
from pathlib import Path
import random

from rave_fl.eval.adapters import read_rows
from rave_fl.eval.schema import CompatibilityError, file_hash


def make_splits(dataset_file, dataset, seed=42, development_ids=("Time-25",)):
    rows, _ = read_rows(dataset_file)
    groups = {}
    ids = []
    for row in rows:
        ident = (
            row.get("instance_id", row.get("bug", ""))
            or f"{row.get('org')}__{row.get('repo')}-{row.get('number')}"
        )
        group = (
            ident.split("-")[0]
            if dataset == "defects4j"
            else row.get("repository", row.get("repo", "custom"))
        )
        if dataset == "multi-swe-bench":
            group = row.get("org", "") + "/" + row.get("repo", "")
        groups.setdefault(group, []).append(ident)
        ids.append(ident)
    if len(set(ids)) != len(ids):
        raise CompatibilityError("Duplicate IDs in split input")
    forced = {k for k, v in groups.items() if set(v) & set(development_ids)}
    remaining = sorted(set(groups) - forced)
    random.Random(seed).shuffle(remaining)
    # Small sets cannot populate every split without sharing projects; empty
    # splits are intentionally preserved and rejected by run if selected.
    labels = ["train", "development", "calibration", "test"]
    out = {k: [] for k in labels}
    for key in sorted(forced):
        out["development"].extend(sorted(groups[key]))
    for index, key in enumerate(remaining):
        out[labels[index % 4]].extend(sorted(groups[key]))
    return (
        out,
        {
            "source_sha256": file_hash(dataset_file),
            "dataset": dataset,
            "seed": seed,
            "group_unit": "Defects4J project or repository",
            "development_ids": list(development_ids),
            "groups": groups,
            "note": "Empty splits are possible; whole projects containing designated development bugs remain development-only",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-file", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--development-ids", default="Time-25")
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise CompatibilityError("Split output exists; preserve prior selection")
    splits, provenance = make_splits(
        args.dataset_file, args.dataset, args.seed, args.development_ids.split(",")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(splits, indent=2) + "\n")
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
