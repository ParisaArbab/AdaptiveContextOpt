"""
fetch_instances.py — WP1

Pulls instances from princeton-nlp/SWE-bench_Lite (HuggingFace) and derives
function-level ground truth (files + function/class names touched) from each
instance's gold patch diff. This ground truth is what agent_localizer.py's
output gets scored against.

Real dataset schema (verified against the live dataset):
    repo, instance_id, base_commit, patch, test_patch, problem_statement,
    hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS,
    environment_setup_commit

Usage:
    python fetch_instances.py --n 15 --seed 42 --out data/instances.json
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


HUNK_HEADER_RE = re.compile(r"^@@ .* @@\s*(.*)$")
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)")
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class GroundTruth:
    instance_id: str
    repo: str
    base_commit: str
    files: List[str]
    functions: List[str]  # "path/to/file.py::func_or_class_name"
    fail_to_pass: List[str]
    problem_statement: str


def parse_gold_patch(patch: str) -> tuple[List[str], List[str]]:
    """Extract touched files and function/class names from a unified diff.

    Two sources of function names, both real signal from the diff itself:
      1. The hunk header's trailing context (`@@ ... @@ def foo(...)`) —
         unified diff format includes the nearest preceding scope line here.
      2. Any `def`/`class` line appearing inside the hunk body (added or
         removed), since a patch that touches a function's internals may not
         always surface it in the hunk header on every hunk.
    """
    files: List[str] = []
    functions: List[str] = []
    current_file = None

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if current_file not in files:
                files.append(current_file)
            continue
        if not current_file:
            continue

        m = HUNK_HEADER_RE.match(line)
        if m and m.group(1):
            scope = m.group(1)
            fm = DEF_RE.match(f" {scope}") or CLASS_RE.match(f" {scope}")
            # hunk header context strips leading whitespace, so also try raw
            fm = fm or re.search(r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", scope)
            if fm:
                name = fm.group(1)
                key = f"{current_file}::{name}"
                if key not in functions:
                    functions.append(key)
            continue

        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            body = line[1:]
            fm = DEF_RE.match(body) or CLASS_RE.match(body)
            if fm:
                key = f"{current_file}::{fm.group(1)}"
                if key not in functions:
                    functions.append(key)

    return files, functions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of instances to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/instances.json")
    ap.add_argument(
        "--instance-ids",
        type=str,
        default="",
        help="comma-separated instance_ids to pin (overrides --n sampling); "
        "use this to reproduce the exact 12/15-instance pilot set from earlier runs",
    )
    args = ap.parse_args()

    from datasets import load_dataset  # imported lazily so --help doesn't need it

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    if args.instance_ids:
        wanted = set(x.strip() for x in args.instance_ids.split(","))
        rows = [r for r in ds if r["instance_id"] in wanted]
    else:
        ds_shuffled = ds.shuffle(seed=args.seed)
        rows = list(ds_shuffled.select(range(min(args.n, len(ds_shuffled)))))

    out: List[dict] = []
    for row in rows:
        files, functions = parse_gold_patch(row["patch"])
        fail_to_pass = row["FAIL_TO_PASS"]
        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass)
        gt = GroundTruth(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            files=files,
            functions=functions,
            fail_to_pass=fail_to_pass,
            problem_statement=row["problem_statement"],
        )
        out.append(asdict(gt))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} instances -> {out_path}")
    for o in out:
        print(f"  {o['instance_id']}: {len(o['files'])} file(s), {len(o['functions'])} function(s)")


if __name__ == "__main__":
    main()
