"""Conservative, arity-aware Java call candidates from saved L1 sketches.

This is *not* a verified Java call graph: Java overload resolution requires
receiver types, conversions, inheritance and a compiler binding pass. It
extracts only explicit 'CALL' entries, checks argument count, and reports
ambiguous/unresolved calls rather than inventing edges. No patch labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CALL = re.compile(r"^CALL\s+(.+)$")
_SUFFIX = re.compile(r"(?:(?P<receiver>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)\(")


@dataclass(frozen=True)
class CallAudit:
    edges: tuple[tuple[int, int], ...]
    ambiguous: tuple[str, ...]
    unresolved: tuple[str, ...]
    inspected_calls: int


def parts(function_id: str) -> tuple[str, str, int]:
    """Return fully qualified owner, method and parameter count."""
    owner_method, suffix = function_id.split("(", 1)
    owner, name = owner_method.rsplit(".", 1)
    params = suffix.rsplit(")", 1)[0]
    return owner, name, count_args(params)


def count_args(args: str) -> int:
    """Count top-level comma-separated arguments, respecting nested calls."""
    if not args.strip():
        return 0
    depth = 0
    angle = 0
    square = 0
    quoted: str | None = None
    escaped = False
    count = 1
    for c in args:
        if quoted is not None:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == quoted:
                quoted = None
            continue
        if c in ("'", '"'):
            quoted = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "[":
            square += 1
        elif c == "]":
            square = max(0, square - 1)
        elif c == "<":
            angle += 1
        elif c == ">":
            angle = max(0, angle - 1)
        elif c == "," and depth == square == angle == 0:
            count += 1
    return count


def _call_signature(call: str) -> tuple[str | None, str, int] | None:
    # Prefer outermost named call; nested calls are recorded on separate
    # lines by the archived L1 sketcher.
    m = _SUFFIX.search(call)
    if not m:
        return None
    start = m.end()
    depth = 1
    end = start
    quoted: str | None = None
    escaped = False
    while end < len(call) and depth:
        c = call[end]
        if quoted is not None:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == quoted:
                quoted = None
        elif c in ("'", '"'):
            quoted = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        end += 1
    if depth:
        return None
    return m.group("receiver"), m.group("name"), count_args(call[start:end-1])


def sketch_call_audit(candidates: list[dict]) -> CallAudit:
    signature = [parts(c["function_id"]) for c in candidates]
    edges = set()
    ambiguous = set()
    unresolved = set()
    inspected = 0
    for caller_idx, candidate in enumerate(candidates):
        caller_owner = signature[caller_idx][0]
        caller_simple_owner = caller_owner.rsplit(".", 1)[-1]
        for line in candidate["levels"][1].splitlines():
            m = _CALL.match(line.strip())
            if not m:
                continue
            inspected += 1
            parsed = _call_signature(m.group(1))
            if parsed is None:
                unresolved.add(f"{caller_idx}:unparsed:{m.group(1)}")
                continue
            receiver, method, arity = parsed
            # Only same-owner unqualified calls, or explicitly named
            # same-owner class. Never infer receiver variable type.
            if receiver not in (None, "this", caller_simple_owner):
                continue
            matches = [j for j, (owner, name, count) in enumerate(signature)
                       if j != caller_idx and owner == caller_owner
                       and name == method and count == arity]
            key = f"{caller_idx}:{method}/{arity}"
            if len(matches) == 1:
                edges.add((caller_idx, matches[0]))
            elif len(matches) > 1:
                ambiguous.add(key)
            else:
                unresolved.add(key)
    return CallAudit(tuple(sorted(edges)), tuple(sorted(ambiguous)),
                     tuple(sorted(unresolved)), inspected)
