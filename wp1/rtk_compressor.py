"""
rtk_compressor.py — WP1 baseline condition (carried forward)

Python re-implementation of rtk's documented heuristics, sourced from its
README, user writeups, and a GitHub issue describing the same failure an
independent user hit. Kept in the pipeline unchanged so the earlier WP1
findings (0.917 -> 0.833 file-level success rate; sympy__sympy-17630 and
sympy__sympy-16792 as confirmed compression-tax instances; the
astropy__astropy-12907 96%-reduction exhibit) remain reproducible and
directly comparable against the new lean-ctx condition.

Known documented behaviors reproduced here:
  - treats an entire pytest FAILURES section as one block rather than
    per-test boundaries
  - applies a fixed result cap on generic-keyword matches (silently drops
    ground-truth files when a keyword returns many hits)
  - compacts repeated assertion diff lines, which can collapse both sides
    of a diff to identical ellipses when they share a common prefix/suffix
    structure
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RESULT_CAP = 20  # documented default match cap
FAILURES_HEADER_RE = re.compile(r"^=+\s*FAILURES\s*=+$", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"^_{5,}|^-{5,}")


@dataclass
class CompressionResult:
    text: str
    mode: str
    original_tokens_est: int
    compressed_tokens_est: int
    reduction_pct: float


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _compact_assertion_diff(lines: list[str]) -> list[str]:
    """Reproduces the documented failure mode: when both sides of an
    assertion diff share long common runs, rtk's line-compaction can reduce
    both to '...' — destroying the one differentiating value."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(("assert ", "AssertionError")) and i + 1 < len(lines):
            # naive compaction: if next line looks like a second value line
            # with >70% shared characters, collapse both
            nxt = lines[i + 1]
            shared = sum(1 for a, b in zip(line, nxt) if a == b)
            if len(line) and shared / max(len(line), 1) > 0.7:
                out.append("...")
                i += 2
                continue
        out.append(line)
        i += 1
    return out


def compress(text: str) -> CompressionResult:
    lines = text.splitlines()
    original_tokens = _estimate_tokens(text)

    # collapse the whole FAILURES section into one block (no per-test split)
    out_lines = []
    in_failures_block = False
    match_count = 0
    for line in lines:
        if FAILURES_HEADER_RE.match(line.strip()):
            in_failures_block = True
            out_lines.append(line)
            continue
        if in_failures_block and SEPARATOR_RE.match(line.strip()):
            continue  # per-test separators dropped: whole section becomes one block
        if in_failures_block:
            match_count += 1
            if match_count > RESULT_CAP:
                continue  # silent truncation past the result cap
        out_lines.append(line)

    out_lines = _compact_assertion_diff(out_lines)
    compressed_text = "\n".join(out_lines)
    compressed_tokens = _estimate_tokens(compressed_text)
    reduction = 100.0 * (1 - compressed_tokens / original_tokens) if original_tokens else 0.0

    return CompressionResult(
        text=compressed_text,
        mode="reference",
        original_tokens_est=original_tokens,
        compressed_tokens_est=compressed_tokens,
        reduction_pct=round(reduction, 1),
    )
