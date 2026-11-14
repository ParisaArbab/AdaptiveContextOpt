"""Explicitly prepare pinned local source checkouts for the preselected smoke bugs.

This is not an official Defects4J checkout command. Commit identity is checked;
a Git commit alone does not certify that it matches Defects4J's buggy revision.
Never change the pinned revision or substitute a bug because of its outcome.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(
        ["git", *(["-C", str(cwd)] if cwd else []), *args],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if r.returncode:
        raise RuntimeError("git " + " ".join(args) + ": " + r.stderr.strip())
    return r.stdout.strip()


def prepare(fetch: bool):
    rows = json.loads(Path("configs/rave_independent_smoke.json").read_text())["cases"]
    for row in rows:
        bug = row["bug"]
        checkout = Path(row["checkout"]).resolve()
        expected = row["buggy_revision"]
        source = row["source_repository"]
        if checkout.exists():
            if not (checkout / ".git").exists():
                raise RuntimeError(f"{bug}: {checkout} exists but is not a standalone Git checkout")
        else:
            if not fetch:
                raise RuntimeError(f"{bug}: missing checkout {checkout}. Rerun with --fetch-pinned.")
            checkout.parent.mkdir(parents=True, exist_ok=True)
            # Clone into a temporary directory first, never expose partial checkouts.
            temp = checkout.with_name(checkout.name + ".preparing")
            if temp.exists():
                raise RuntimeError(f"{bug}: stale preparation directory {temp}; inspect manually")
            try:
                git("clone", "--no-checkout", "--filter=blob:none", source, str(temp))
                git("checkout", "--detach", expected, cwd=temp)
                actual = git("rev-parse", "HEAD", cwd=temp)
                if actual != expected:
                    raise RuntimeError(f"{bug}: wrong revision {actual}, expected {expected}")
                if git("status", "--porcelain", cwd=temp):
                    raise RuntimeError(f"{bug}: prepared checkout has modifications")
                temp.rename(checkout)
            except BaseException:
                # Preserve a failed download for inspection, do not silently replace it.
                raise
        actual = git("rev-parse", "HEAD", cwd=checkout)
        if actual != expected:
            raise RuntimeError(f"{bug}: checkout SHA {actual} does not match recorded {expected}")
        if git("status", "--porcelain", cwd=checkout):
            raise RuntimeError(f"{bug}: checkout is dirty; use a clean pinned checkout")
        print(f"CHECKOUT READY: {bug} {actual} {checkout}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetch-pinned", action="store_true",
        help="Explicitly clone missing repositories and checkout only recorded SHAs"
    )
    args = parser.parse_args()
    prepare(args.fetch_pinned)
