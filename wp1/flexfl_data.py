"""Readers for the original FlexFL replication data used by this project."""
from __future__ import annotations

import csv
import json
from pathlib import Path

TRADITIONAL_METHODS = ("SBIR", "Ochiai", "BoostN")


def flexfl_root(repo: Path) -> Path:
    repo = Path(repo)
    nested = repo / "FlexFL"
    return nested if nested.is_dir() else repo


def read_text_input(repo: Path, category: str, bug: str) -> str:
    root = flexfl_root(repo)
    base = root / "data" / "input" / category
    candidates = [
        base / "Defects4J" / f"{bug}.txt",
        base / f"{bug}.txt",
        base / "Defects4J" / bug,
        base / bug,
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(errors="replace")
    return ""


def read_bug_report(repo: Path, bug: str) -> str:
    return read_text_input(repo, "bug_reports", bug)


def read_trigger_test(repo: Path, bug: str) -> str:
    return read_text_input(repo, "trigger_tests", bug)


def read_ground_truth(repo: Path, bug: str) -> list[str]:
    root = flexfl_root(repo)
    path = root / "data" / "input" / "ground_truth" / "Defects4J" / "gt.json"
    if not path.is_file():
        raise FileNotFoundError(f"FlexFL ground truth not found: {path}")
    data = json.loads(path.read_text())
    value = data.get(bug, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def read_traditional_top5(repo: Path, method: str, bug: str) -> list[str]:
    if method not in TRADITIONAL_METHODS:
        raise ValueError(f"Unknown FL method {method!r}, expected one of {TRADITIONAL_METHODS}")
    root = flexfl_root(repo)
    path = root / "data" / "FL_results" / method / "Defects4J" / f"{bug}_method-susps.csv"
    if not path.is_file():
        raise FileNotFoundError(f"FlexFL {method} result missing for {bug}: {path}")
    out: list[str] = []
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = (row.get("File") or row.get("file") or "").strip()
            signature = (row.get("Signature") or row.get("signature") or "").strip()
            if file_name and signature:
                out.append(f"{file_name}.{signature}")
            if len(out) == 5:
                break
    return out


def merge_top20(repo: Path, bug: str, agent4sr_top5: list[str]) -> tuple[list[str], dict]:
    """Match FlexFL combine.py: 5 SBIR + 5 Ochiai + 5 BoostN + 5 Agent4SR.

    The original implementation appends lists and does not deduplicate them. We keep
    that behavior because changing it would change the candidate distribution.
    """
    parts = {method: read_traditional_top5(repo, method, bug) for method in TRADITIONAL_METHODS}
    parts["Agent4SR"] = list(agent4sr_top5[:5])
    merged: list[str] = []
    for method in (*TRADITIONAL_METHODS, "Agent4SR"):
        merged.extend(parts[method])
    return merged[:20], parts
