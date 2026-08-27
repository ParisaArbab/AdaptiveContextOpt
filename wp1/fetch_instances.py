"""
fetch_instances.py — WP1

Pulls instances from any SWE-bench-shaped dataset and derives ground truth
(files + function/class symbols touched) from each instance's gold patch.

Dataset and language are no longer hardcoded: `--dataset` takes a registry
alias (swe-bench-lite / -verified / swe-bench / swe-bench-java / ...) or a
raw HuggingFace id, and the matching LanguageAdapter in benchmarks.py owns
patch parsing. `--field-map logical=column` handles a fork whose schema
differs without touching code.

The `test_patch` is now carried through to the harness (it holds the
FAIL_TO_PASS test itself — without it the trigger test doesn't exist in the
checkout), as is `fail_to_pass`, which selects exactly which tests run.

Usage:
    python fetch_instances.py --dataset swe-bench-lite --n 15 --seed 42 \
        --out data/instances.json
    python fetch_instances.py --dataset swe-bench-java --language java \
        --n 15 --out data/java_instances.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

from benchmarks import coerce_list, parse_field_map_args, resolve_dataset, LANGUAGES


@dataclass
class GroundTruth:
    instance_id: str
    repo: str
    base_commit: str
    files: List[str]
    functions: List[str]          # "path/to/file.ext::SymbolName"
    fail_to_pass: List[str]
    problem_statement: str
    test_patch: str
    dataset: str
    language: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swe-bench-lite",
                    help="registry alias or raw HuggingFace dataset id")
    ap.add_argument("--language", default=None, choices=sorted(LANGUAGES) + [None],
                    help="override the registry's language for this dataset")
    ap.add_argument("--split", default=None)
    ap.add_argument("--field-map", action="append", default=[],
                    help="logical=column override, repeatable (e.g. fail_to_pass=F2P)")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/instances.json")
    ap.add_argument("--instance-ids", default="",
                    help="comma-separated instance_ids to pin (overrides --n sampling)")
    args = ap.parse_args()

    spec = resolve_dataset(args.dataset, args.language, parse_field_map_args(args.field_map))
    if args.split:
        spec.split = args.split
    if not spec.verified:
        print(f"note: {args.dataset} schema is not confirmed in this repo"
              + (f" — {spec.note}" if spec.note else ""))
    adapter = spec.adapter

    from datasets import load_dataset  # lazy so --help works without it

    ds = load_dataset(spec.hf_id, split=spec.split)

    if args.instance_ids:
        wanted = {x.strip() for x in args.instance_ids.split(",")}
        rows = [r for r in ds if spec.get(r, "instance_id") in wanted]
    else:
        rows = list(ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds)))))

    out: List[dict] = []
    for row in rows:
        files, functions = adapter.parse_patch_symbols(spec.get(row, "patch", "") or "")
        gt = GroundTruth(
            instance_id=spec.get(row, "instance_id"),
            repo=spec.get(row, "repo"),
            base_commit=spec.get(row, "base_commit"),
            files=files,
            functions=functions,
            fail_to_pass=coerce_list(spec.get(row, "fail_to_pass")),
            problem_statement=spec.get(row, "problem_statement", "") or "",
            test_patch=spec.get(row, "test_patch", "") or "",
            dataset=args.dataset,
            language=spec.language,
        )
        out.append(asdict(gt))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} instances ({args.dataset}/{spec.language}) -> {out_path}")

    missing_syms = sum(1 for o in out if not o["functions"])
    missing_tests = sum(1 for o in out if not o["fail_to_pass"])
    for o in out:
        print(f"  {o['instance_id']}: {len(o['files'])} file(s), "
              f"{len(o['functions'])} symbol(s), {len(o['fail_to_pass'])} trigger test(s)")
    if missing_syms:
        print(f"warning: {missing_syms} instance(s) yielded no symbols — those are "
              f"excluded from method-level scoring (file-level still applies)")
    if missing_tests:
        print(f"warning: {missing_tests} instance(s) have no FAIL_TO_PASS ids")


if __name__ == "__main__":
    main()
