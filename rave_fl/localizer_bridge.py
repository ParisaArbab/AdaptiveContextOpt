"""Optional live base-localizer adapter.

A manifest can supply `localizer_command` (an argv list, never a shell string).
The command must output exactly one JSON object on stdout, containing ranked
candidate methods with *measured* semantic and coverage scores, and their
source spans. This adapter intentionally does NOT pretend that SBIR ordinal
priors are semantic similarity or real coverage observations.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from rave_fl.core import Candidate


def _score(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a real normalized score between 0 and 1")
    return float(value)


def run_localizer(case: dict, timeout: int = 600):
    """Return (candidates, source span metadata, provenance), without gold labels.

    Example case entry:
      "localizer_command": ["python", "-m", "my_localizer",
                            "--bug", "{bug}", "--checkout", "{checkout}"]
    Command output schema:
      {"localizer": "my-localizer-v1",
       "runtime_pruning": "measured" | "not_available",
       "candidates": [{
          "function_id": "package.Class.method(int)",
          "file_class": "package.Class", "start_line": 10, "end_line": 30,
          "semantic_score": 0.72, "coverage_score": 0.31,
          "ranker_scores": {"retrieval": 0.72, "coverage": 0.31}}]}
    Source spans must refer to the buggy checkout, not a patched revision.
    """
    argv = case.get("localizer_command")
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
        raise ValueError("localizer_command must be a nonempty argv list; shell commands are not accepted")
    fields = {k: str(case.get(k, "")) for k in ("bug", "checkout", "evidence_file", "flexfl_repo")}
    try:
        command = [s.format_map(fields) for s in argv]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid localizer_command placeholder: {exc}") from exc
    proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f"Base localizer failed with exit {proc.returncode}: {proc.stderr[-1000:]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Base localizer must print a single JSON object to stdout") from exc
    if not isinstance(data, dict) or not isinstance(data.get("localizer"), str) or not data["localizer"].strip():
        raise ValueError("Localizer output needs a nonempty localizer identifier")
    if data.get("runtime_pruning") not in {"measured", "not_available"}:
        raise ValueError("runtime_pruning must state 'measured' or 'not_available'")
    rows = data.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Localizer returned no candidate methods")
    candidates, spans, ids = [], {}, set()
    for order, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("Candidate entries must be objects")
        ident = row.get("function_id")
        cls = row.get("file_class")
        start, end = row.get("start_line"), row.get("end_line")
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError("Candidate identifiers must be unique nonempty strings")
        if not isinstance(cls, str) or not cls.strip():
            raise ValueError(f"{ident}: file_class is missing")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or not 0 < start <= end:
            raise ValueError(f"{ident}: invalid buggy-source line span")
        sem = _score(row.get("semantic_score"), "semantic_score")
        cov = _score(row.get("coverage_score"), "coverage_score")
        signals = row.get("ranker_scores", {})
        if not isinstance(signals, dict):
            raise ValueError(f"{ident}: ranker_scores must be a mapping")
        for key, val in signals.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Invalid ranker name")
            _score(val, f"{key} ranker score")
        ids.add(ident)
        # Input order is the localizer's explicit ranking, not ground-truth order.
        candidates.append(Candidate(ident, ident, semantic_score=sem, coverage_score=cov))
        spans[ident] = {
            "file_class": cls, "start_line": start, "end_line": end,
            "methods": list(signals), "ranks": {}, "top_per_method": len(rows),
            "semantic_score": sem, "coverage_score": cov,
            "prior": (sem + cov) / 2, "ranker_scores": signals,
            "localizer_rank": order,
        }
    return candidates, spans, {
        "mode": "live_external", "name": data["localizer"],
        "candidate_count": len(candidates), "runtime_pruning": data["runtime_pruning"],
        "command_argv": command,
        "measurement_note": "Only scores supplied by the live localizer are considered measured.",
    }
