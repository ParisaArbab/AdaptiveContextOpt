"""Publish the existing AIScalar corrected-BM25 recovery artifacts to GitHub.

Runs on AIScalar, where the artifacts actually exist. Includes every regular
file under each completed/attempted batch's experiment directory, plus the
recovery protocol. Does not follow symlinks, upload prepared checkouts, or
modify archived model results. Rejects secrets and oversized files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess

ROOT = Path("results/graft_100_bm25_recovery_v1")
BATCHES = (2, 6, 7, 11, 14, 19)
MAX_FILE_SIZE = 90 * 1024 * 1024
# Stop, rather than silently publishing possible secrets.
SECRETS = [
    re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"sk-[A-Za-z0-9]{24,}"),
    re.compile(rb"-----BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY-----"),
    re.compile(rb"(?i)(?:authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/-]{15,}"),
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def candidates() -> list[Path]:
    if not (ROOT / "recovery_protocol.json").is_file():
        raise RuntimeError("Recovery protocol missing; refusing to archive a different run")
    files = [ROOT / "recovery_protocol.json"]
    for batch in BATCHES:
        exp = ROOT / "batches" / f"batch_{batch:02d}" / "experiment"
        if not exp.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(exp, followlinks=False):
            dirnames[:] = [name for name in dirnames
                           if not (Path(dirpath) / name).is_symlink()]
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_symlink() or name.endswith(".lock") or not path.is_file():
                    continue
                files.append(path)
    return sorted(files)


def inspect(files: list[Path]) -> int:
    size = 0
    for path in files:
        length = path.stat().st_size
        size += length
        if length > MAX_FILE_SIZE:
            raise RuntimeError(f"Oversized file ({length} bytes): {path}. "
                               "Use Git LFS or a separate archive; nothing was staged.")
        # Scan in chunks with a small overlap for tokens across chunk boundaries.
        # These files can contain failure evidence and LLM prompts: review
        # privacy and licenses before running with --push.
        tail = b""
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                text = tail + chunk
                if any(pattern.search(text) for pattern in SECRETS):
                    raise RuntimeError(f"Possible credential in {path}; nothing was staged")
                tail = text[-512:]
    return size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--push", action="store_true",
                   help="Explicitly stage, commit, and push the inspected files")
    args = p.parse_args()
    branch = "graft-fl-recovery-analysis-20261116"
    if git("branch", "--show-current") != branch:
        raise RuntimeError(f"Switch to {branch} before publishing")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked working tree must be clean before publishing")
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("Staging area is not empty; preserve other changes")
    remote = git("remote", "get-url", "origin")
    if "ParisaArbab/AdaptiveContextOpt" not in remote:
        raise RuntimeError(f"Unexpected origin: {remote}")
    files = candidates()
    total_bytes = inspect(files)
    print(f"Inspected {len(files)} files, {total_bytes:,} bytes.")
    for batch in BATCHES:
        subset = [f for f in files if f"batch_{batch:02d}" in f.parts]
        print(f"Batch {batch:02d}: {len(subset)} artifact files")
    print("Includes: experiment configs, logs, metrics, ledgers, cache, "
          "completion gates, and accounting where present.")
    print("Excludes: lock files, checkouts, symlinks, and nonexistent batches.")
    if not args.push:
        print("Dry run only. Inspect for private data, then rerun with --push.")
        return
    subprocess.run(["git", "add", "-f", "--", *map(str, files)], check=True)
    if not git("diff", "--cached", "--name-only"):
        print("No new changes to commit.")
        return
    # Do not allow unsafe metadata in staged content.
    subprocess.run(["git", "-c", "core.whitespace=-blank-at-eol,-space-before-tab",
                    "diff", "--cached", "--check"], check=True)
    subprocess.run(["git", "-c", "user.name=ParisaArbab",
                    "-c", "user.email=pa.arbab@gmail.com",
                    "commit", "-m",
                    "experiment(graft): archive corrected BM25 recovery results and ledgers"],
                   check=True)
    subprocess.run(["git", "push", "origin", branch], check=True)
    print("Pushed available recovery artifacts to", branch)


if __name__ == "__main__":
    main()
