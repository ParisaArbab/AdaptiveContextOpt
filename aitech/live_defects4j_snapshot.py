"""Build a pre-gold L0/L1 source snapshot from a real Defects4J checkout.

This script runs failing tests in an EXISTING buggy checkout. It never
reads patches, classes.modified, or fixed revisions. Retrieval ranks
methods from failure text and Java source in the buggy checkout.

Java declaration extraction is an explicitly limited regex/brace
heuristic, NOT a compiler-resolved method identity or source slice.
It produces a reproducible exploratory pool, not an ARISE or FaR-Loc
reimplementation. Empty failure output is an error, not a success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from .defects4j_compiler_pilot import export

FRAME = re.compile(r"\bat\s+([\w.$]+)\.([\w$<>]+)\(([^():]+\.java):(\d+)\)")
DECL = re.compile(
    r"(?m)^\s*(?:(?:public|private|protected|static|final|abstract|"
    r"synchronized|native|default|strictfp)\s+)*"
    r"(?:[\w.$<>\[\],?]+\s+)?([a-zA-Z_$][\w$]*)\s*"
    r"\(([^;{}()]*)\)\s*(?:throws\s+[\w.,\s]+)?\s*\{"
)
CONTROL = {"if", "for", "while", "switch", "catch", "new", "return", "throw", "try"}


def run_failed_tests(checkout: Path, timeout: int = 300) -> str:
    completed = subprocess.run(
        ["defects4j", "test"], cwd=checkout, capture_output=True,
        text=True, timeout=timeout, check=False)
    combined = completed.stdout + "\n" + completed.stderr
    if not combined.strip() or (
        "Failing tests:" not in combined and "FAIL" not in combined
        and "Exception" not in combined and "Error" not in combined):
        raise RuntimeError("defects4j test did not provide failure evidence")
    match = re.search(r"Failing tests:\\s*(\\d+)", combined)
    if match is not None and int(match.group(1)) == 0:
        raise RuntimeError("buggy checkout reported zero failing tests")
    return combined[-25000:]


def _methods(source: str, filename: str, owner: str) -> list[dict]:
    found = []
    for match in DECL.finditer(source):
        name = match.group(1)
        if name in CONTROL:
            continue
        # Bracket counting is approximate for strings/comments. Reject
        # unclosed snippets and expose exact source provenance.
        balance, end = 1, match.end()
        while end < len(source) and balance:
            if source[end] == "{":
                balance += 1
            elif source[end] == "}":
                balance -= 1
            end += 1
        if balance:
            continue
        start = match.start()
        line = source.count("\n", 0, start) + 1
        parameter_text = match.group(2).strip()
        raw_params = [s.strip() for s in parameter_text.split(",") if s.strip()]
        params = []
        for parameter in raw_params:
            tokens = [t for t in parameter.split() if not t.startswith("@")]
            params.append(tokens[-2] if len(tokens) >= 2 else
                          (tokens[0] if tokens else "?"))
        function_id = f"{owner}.{name}({','.join(params)})"
        detail = source[start:min(end, start + 1200)].strip()
        found.append({
            "function_id": function_id,
            "levels": [
                f"FILE {filename} LINE {line}: {function_id}",
                f"FILE {filename} LINE {line}: {function_id}\n{detail}",
            ],
            "source_path": filename, "source_line": line,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "extraction": "java_regex_brace_heuristic_unverified",
        })
    return found


def compact_failure_text(text: str, max_chars: int = 2600) -> str:
    """Preserve failed-test headline and early trace frames under a fixed cap."""
    if max_chars < 500:
        raise ValueError("failure context budget too small")
    lines = text.splitlines()
    selected = []
    for line in lines:
        if ("Failing tests:" in line or "::" in line or "Exception" in line
                or "Error" in line or "Assertion" in line
                or FRAME.search(line) is not None):
            selected.append(line)
    if not selected:
        selected = lines[:30]
    return "\n".join(selected)[:max_chars]


def extract_snapshot(checkout: Path, failure_text: str, *, limit: int = 25,
                     max_files: int = 2000) -> dict:
    if limit < 2 or max_files < 1:
        raise ValueError("invalid extraction limits")
    checkout = checkout.resolve()
    if not (checkout / ".defects4j.config").is_file():
        raise ValueError("requires real buggy Defects4J checkout")
    src_dir = (checkout / export(checkout, "dir.src.classes")).resolve()
    if not src_dir.is_relative_to(checkout) or not src_dir.is_dir():
        raise ValueError("invalid source directory")
    paths = sorted(src_dir.rglob("*.java"))
    if len(paths) > max_files:
        raise RuntimeError("too many sources, no silent partial scan")
    compact_context = compact_failure_text(failure_text)
    frame_hits = FRAME.findall(compact_context)
    frame_classes = {x[0].replace("$", ".") for x in frame_hits}
    frame_methods = {x[1] for x in frame_hits}
    failure_terms = set(re.findall(r"[a-zA-Z]{4,}", compact_context.lower()))
    methods = []
    for file in paths:
        raw = file.read_bytes()
        source = raw.decode("utf-8", errors="replace")
        package = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", source)
        owner = (package.group(1) + "." if package else "") + file.stem
        relative = str(file.relative_to(checkout))
        for item in _methods(source, relative, owner):
            method = item["function_id"].split("(", 1)[0].rsplit(".", 1)[-1]
            score = 0
            if owner in frame_classes:
                score += 30
            if method in frame_methods:
                score += 12
            score += len(failure_terms.intersection(
                set(re.findall(r"[a-zA-Z]{4,}", item["levels"][1].lower()))))
            item["retrieval_score_proxy"] = score
            methods.append(item)
    methods.sort(key=lambda x: (-x["retrieval_score_proxy"], x["function_id"],
                                x["source_path"], x["source_line"]))
    candidates = []
    seen = set()
    for item in methods:
        key = item["function_id"]
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)
        if len(candidates) >= limit:
            break
    if not candidates:
        raise RuntimeError("no source methods extracted")
    snapshot = {
        "context": compact_context,
        "candidates": [
            {
                **item,
                "prior": max(0.0, min(1.0, item["retrieval_score_proxy"] / 50)),
                "features": {
                    "l0_tokens": len(item["levels"][0].split()),
                    "l1_tokens": len(item["levels"][1].split()),
                    "failure_relevance": max(
                        0.0, min(1.0, item["retrieval_score_proxy"] / 50)),
                    "trace_match": float(item["retrieval_score_proxy"] >= 30),
                    "ranker_agreement": 0.0,
                },
            } for item in candidates
        ],
        "provenance": {
            "source_root": str(src_dir),
            "source_file_count": len(paths),
            "frame_count": len(frame_hits),
            "raw_failure_characters": len(failure_text),
            "compact_failure_characters": len(compact_context),
            "failure_sha256": hashlib.sha256(
                failure_text.encode("utf-8")).hexdigest(),
            "candidate_extraction": "regex heuristic, not compiler-verified",
            "gold_patches_loaded": False,
        },
    }
    return snapshot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--failure-output", type=Path)
    p.add_argument("--limit", type=int, default=25)
    args = p.parse_args()
    if args.failure_output:
        failure = args.failure_output.read_text(encoding="utf-8")
    else:
        failure = run_failed_tests(args.checkout)
    result = extract_snapshot(args.checkout, failure, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(result['candidates'])} pre-gold candidate methods")


if __name__ == "__main__":
    main()
