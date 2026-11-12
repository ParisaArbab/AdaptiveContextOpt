"""Validate fault-evidence support annotations before training.

The goal is to collect independently audited support-path outcomes from
real buggy repository revisions. The schema is strict and rejects
cross-project split leakage; this module neither invents labels nor
trains a model. An annotation's sufficient=True is a human/auditor
judgment to test, not ground truth merely because a call exists.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter


REQUIRED = {
    "bug", "project", "buggy_revision", "anchor_id", "support_ids",
    "provenance", "compiler_binding_verified", "sufficient",
    "audit_reference", "split",
}
SPLITS = {"train", "calibration", "test"}


def validate_annotations(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no annotations supplied")
    issues: list[str] = []
    project_split: dict[str, str] = {}
    seen: set[tuple] = set()
    class_balance = Counter()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != REQUIRED:
            issues.append(f"row {i}: schema mismatch")
            continue
        if row["split"] not in SPLITS:
            issues.append(f"row {i}: unknown split")
            continue
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ("bug", "project", "buggy_revision", "anchor_id",
                           "provenance", "audit_reference")):
            issues.append(f"row {i}: missing provenance or audit reference")
        support_ids = row["support_ids"]
        if (not isinstance(support_ids, list) or not support_ids
                or any(not isinstance(x, str) or not x.strip() for x in support_ids)
                or len(set(support_ids)) != len(support_ids)):
            issues.append(f"row {i}: invalid support bundle")
            continue
        if row["anchor_id"] in support_ids:
            issues.append(f"row {i}: anchor cannot support itself")
        if type(row["compiler_binding_verified"]) is not bool or type(row["sufficient"]) is not bool:
            issues.append(f"row {i}: labels must be explicit booleans")
        if row["sufficient"] and not row["compiler_binding_verified"]:
            issues.append(f"row {i}: sufficient call-based path needs verified binding")
        old = project_split.setdefault(row["project"], row["split"])
        if old != row["split"]:
            issues.append(f"row {i}: project-level split leakage ({row['project']})")
        key = (row["bug"], row["anchor_id"], tuple(support_ids))
        if key in seen:
            issues.append(f"row {i}: repeated annotation key")
        seen.add(key)
        if type(row["sufficient"]) is bool:
            class_balance[(row["split"], row["sufficient"])] += 1
    if issues:
        raise ValueError("; ".join(issues))
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "annotation_count": len(rows),
        "project_count": len(project_split),
        "splits": {split: sum(row["split"] == split for row in rows)
                   for split in sorted(SPLITS)},
        "positive_counts": {split: class_balance[(split, True)]
                            for split in sorted(SPLITS)},
        "annotations_sha256": digest,
        "labels_are_independently_audited": "requires external verification, not checked by this schema",
        "trained_model": False,
    }
