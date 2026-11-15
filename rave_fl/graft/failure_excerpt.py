"""Auditable bounded excerpts of an observed Defects4J failure.

The graph retains the entire immutable failure node. This helper selects only
EXACT source lines, with their original line numbers and full source digest.
It never paraphrases exceptions, infers test outcomes, or reads gold labels.
This is an explicit opt-in Phase-3 prompt-budget condition, not the v1 method.
"""
from __future__ import annotations

from hashlib import sha256
import re


SCHEMA = "graft-failure-line-excerpt-v1"
MAX_SOURCE_BYTES = 1650
IDENTITY_PREFIXES = (
    "OFFICIAL DEFECTS4J BUG:",
    "EXECUTED BUGGY TEST:",
    "TEST SOURCE:",
    "TEST SOURCE SHA256:",
)
TEST_HEADER = "COMPLETE EXECUTED TEST METHOD (verbatim):"
OBSERVED_HEADER = "OBSERVED DEFECTS4J FAILING_TESTS OUTPUT (verbatim):"


def bounded_failure_excerpt(full: str) -> dict:
    """Return deterministic, exact UTF-8 source lines and visible omissions.

    Required fields and at least one observed exception are mandatory. This
    routine fails closed for a changed/unrecognized evidence format, rather than
    showing a manufactured test summary.
    """
    if not isinstance(full, str) or not full:
        raise ValueError("Nonempty real failure evidence required")
    lines = full.splitlines()
    identity = [
        i for i, line in enumerate(lines)
        if any(line.startswith(prefix) for prefix in IDENTITY_PREFIXES)
    ]
    if len(identity) != len(IDENTITY_PREFIXES):
        raise ValueError("Unknown real Defects4J evidence header layout")
    try:
        begin = lines.index(TEST_HEADER)
        end = lines.index(OBSERVED_HEADER)
    except ValueError as exc:
        raise ValueError("Missing executed test or observed output boundary") from exc
    if not begin < end:
        raise ValueError("Executed test must precede observed failures")

    # Select explicit identity, the actual method header, observed test markers,
    # exception messages and a few original (unmodified) top stack frames.
    obligatory = [*identity, begin, end]
    observed = list(range(end + 1, len(lines)))
    groups = []
    group = []
    for i in observed:
        if lines[i].startswith("--- "):
            if group:
                groups.append(group)
            group = [i]
        elif group:
            group.append(i)
    if group:
        groups.append(group)
    if not groups:
        raise ValueError("No observed failed test markers in failure source")

    for block in groups[:2]:
        obligatory.append(block[0])
        exception = next(
            (i for i in block[1:] if lines[i].strip() and not lines[i].lstrip().startswith("at ")),
            None,
        )
        if exception is None:
            raise ValueError("Observed failed test has no exception message")
        obligatory.append(exception)
        top = [i for i in block if lines[i].lstrip().startswith("at ")]
        obligatory.extend(top[:2])

    # Include the start/end of the real method (not generated prose). The test
    # can be long; omitted sections are reported, not silently stitched.
    test_lines = list(range(begin + 1, end))
    relevant = [
        i for i in test_lines if re.search(
            r"\b(?:assert\w*|fail|throw|catch|new\s+\w+)\s*\(", lines[i]
        )
    ]
    suggestions = (
        test_lines[:3]
        + relevant[-5:]
        + test_lines[-4:]
    )
    chosen = set()
    used = 0
    for index in list(dict.fromkeys(obligatory + suggestions)):
        raw = lines[index].encode("utf-8")
        if used + len(raw) + 1 > MAX_SOURCE_BYTES:
            if index in obligatory:
                raise ValueError("Observed mandatory failure lines exceed excerpt bound")
            continue
        chosen.add(index)
        used += len(raw) + 1

    included = sorted(chosen)
    if not included or not any(i > end for i in included):
        raise ValueError("Failed to retain observed exception evidence")
    return {
        "schema": SCHEMA,
        "source_sha256": sha256(full.encode("utf-8")).hexdigest(),
        "original_line_count": len(lines),
        "original_utf8_bytes": len(full.encode("utf-8")),
        "selection_rule": "identity_test_head_tail_assertions_first_two_observed_failures_and_frames",
        "source_line_bytes_limit": MAX_SOURCE_BYTES,
        "excerpt_is_incomplete": len(included) != len(lines),
        "selected_original_lines": [[i + 1, lines[i]] for i in included],
        "note": "Exact lines only. Unlisted original source lines were omitted, not checked or found irrelevant.",
    }
