"""Run compiler-attributed call extraction on a REAL Defects4J buggy checkout.

No implicit checkout, deletion, network access, or benchmark labels.
Use a pre-existing `defects4j checkout -p Lang -v 1b -w ...` tree.

The harness exports class and classpath metadata from the same buggy checkout,
freezes source SHA-256 identities, and fails when Java attribution is incomplete.
Compiler output is *static targets*, not runtime polymorphic dispatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .java_compiler_graph import audit_source_directory


def export(checkout: Path, key: str) -> str:
    proc = subprocess.run(
        ["defects4j", "export", "-p", key],
        cwd=checkout, capture_output=True, text=True, timeout=90, check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"defects4j export {key} failed: {proc.stderr[-500:]}")
    return proc.stdout.strip()


def resolve_metadata(checkout: Path) -> dict:
    checkout = checkout.resolve()
    if not (checkout / ".defects4j.config").is_file():
        raise ValueError("Requires an existing, valid Defects4J checkout")
    classes = export(checkout, "dir.src.classes")
    root = (checkout / classes).resolve()
    if not root.is_relative_to(checkout) or not root.is_dir():
        raise ValueError("Exported source directory is missing or outside checkout")
    raw_classpath = export(checkout, "cp.compile")
    entries = []
    for piece in raw_classpath.split(os.pathsep):
        if not piece.strip():
            continue
        p = Path(piece)
        entries.append(str(p if p.is_absolute() else checkout / p))
    return {
        "checkout": str(checkout),
        "source_directory": str(root),
        "compile_classpath": os.pathsep.join(entries),
        "compile_classpath_entries": len(entries),
        "defects4j_metadata_sha256": hashlib.sha256(
            (checkout / ".defects4j.config").read_bytes()
        ).hexdigest(),
    }


def run(checkout: Path, *, timeout: int = 240, max_files: int = 300) -> dict:
    info = resolve_metadata(checkout)
    source = Path(info["source_directory"])
    count = sum(1 for _ in source.rglob("*.java"))
    if count > max_files:
        raise RuntimeError(
            f"Source set has {count} files, exceeds {max_files} file limit. "
            "No partial attribution was performed, increase --max-files only "
            "when CPU/memory allow."
        )
    audit = audit_source_directory(
        source, classpath=info["compile_classpath"], timeout=timeout)
    return {
        "status": audit.status,
        "checkout": info["checkout"],
        "source_directory": info["source_directory"],
        "source_count": audit.source_files,
        "file_limit": max_files,
        "classpath_entries": info["compile_classpath_entries"],
        "defects4j_metadata_sha256": info["defects4j_metadata_sha256"],
        "source_hashes": audit.source_hashes,
        "resolved_static_calls": len(audit.calls),
        "unresolved_calls": audit.unresolved_count,
        "calls": [x.__dict__ for x in audit.calls],
        "gold_patch_used": False,
        "limitations": [
            "Compiler classpath must be complete and compatible with current JDK",
            "Only compilation-successful source sets are accepted",
            "JDK static targets do not account for runtime dispatch or reflection",
            "No LLM inference or fault-localization accuracy measurement",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-files", type=int, default=300)
    args = parser.parse_args()
    result = run(args.checkout, timeout=args.timeout, max_files=args.max_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {result['resolved_static_calls']} static call targets "
          f"from {result['source_count']} real source files to {args.output}")


if __name__ == "__main__":
    main()
