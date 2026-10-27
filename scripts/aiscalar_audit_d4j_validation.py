#!/usr/bin/env python3
"""Audit prepared/failed Defects4J source validation without changing a checkout.

Emit a new, frozen manifest containing only successfully source-validated bugs.
Keep the original 100 selected/attempted denominator and reasons in an audit JSON.
Never modify, restore, or replace git tracked sources. No LLM calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def git_status(path: Path) -> dict:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain",
         "--untracked-files=no"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode:
        return {"error": result.stderr.strip()}
    changes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"tracked_changes": changes}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="configs/aiscalar_d4j_100.json")
    p.add_argument("--validation", default="results/d4j_100_validation")
    p.add_argument("--output", default="configs/aiscalar_d4j_validated.json")
    p.add_argument("--audit", default="logs/d4j_100_validation_audit.json")
    args = p.parse_args()
    source, validation = Path(args.source), Path(args.validation)
    manifest = read_json(source)
    if manifest.get("path_base") != "cwd":
        p.error("Expected cwd-based manifest. Validate path semantics before filtering.")
    rows = manifest.get("instances")
    if not isinstance(rows, list) or not rows:
        p.error("No instances in source manifest")
    ids = [r.get("instance_id", "") for r in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        p.error("Missing or duplicate source instance IDs")
    statuses = {}
    for path in (validation / "instances").glob("*/status.json"):
        info = read_json(path)
        ident = info.get("instance_id")
        if not ident or ident in statuses:
            p.error("Missing or duplicate instance_id in validation statuses")
        statuses[ident] = {
            "status": info.get("status", "unknown"),
            "error": info.get("error", ""),
            "status_file": str(path),
        }
    extra = set(statuses) - set(ids)
    missing = set(ids) - set(statuses)
    if extra or missing:
        p.error(
            f"Validation/source mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}"
        )

    kept = []
    failed = {}
    for row in rows:
        ident = row["instance_id"]
        status = statuses[ident]
        if status["status"] == "prepared":
            # Reconfirm the checkout's tracked files are unchanged and the HEAD
            # is the expected base before committing to this filtered manifest.
            checkout = Path(row["checkout"]).resolve()
            cur = git_status(checkout)
            if cur.get("error") or cur.get("tracked_changes"):
                failed[ident] = {
                    "status": "changed_since_validation",
                    "error": str(cur),
                    "checkout": str(checkout)
                }
                continue
            head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=30)
            if head.returncode or head.stdout.strip() != row.get("revision"):
                failed[ident] = {
                    "status": "revision_mismatch",
                    "error": head.stderr.strip() or "HEAD does not equal manifest revision",
                    "checkout": str(checkout),
                }
                continue
            kept.append(row)
        else:
            info = dict(status)
            checkout = Path(row["checkout"]).resolve()
            if "Tracked checkout files modified" in status["error"] and checkout.is_dir():
                info.update(git_status(checkout))
            failed[ident] = info

    filtered = {
        **manifest,
        "version": str(manifest.get("version", "local")) + "-validated",
        "instances": kept,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != filtered:
        p.error(f"Output differs from existing manifest, choose new --output: {output}")
    output.write_text(json.dumps(filtered, indent=2) + "\n")
    audit = {
        "selected_attempted": len(ids),
        "source_manifest": str(source.resolve()),
        "validation_directory": str(validation.resolve()),
        "validated_for_inference": len(kept),
        "excluded": len(failed),
        "failed": failed,
        "kept_bug_ids": [r["instance_id"] for r in kept],
        "exclusion_denominator_note": (
            "Failures remain in the selected/attempted denominator. Conditional "
            "metrics on validated bugs must be reported separately."
        ),
    }
    audit_file = Path(args.audit)
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Selected: {len(ids)}. Validated: {len(kept)}. Excluded: {len(failed)}.")
    print("Filtered manifest:", output)
    print("Audit:", audit_file)
    for bug, detail in failed.items():
        print(f"  {bug}: {detail.get('status')}: {detail.get('error')}")
        if detail.get("tracked_changes"):
            for entry in detail["tracked_changes"]:
                print(f"    tracked: {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
