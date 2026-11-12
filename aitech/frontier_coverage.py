"""Audit missing class-qualified Java calls in a ranked code evidence pool.

This reads explicit CALL lines from archived method sketches. It reports
which references point to unique in-pool candidates and which likely
require additional code retrieval. No Java type binding is performed;
all accepted relations are lexical candidates with source provenance,
not verified static or runtime call edges.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .java_call_audit import _call_signature, parts


@dataclass(frozen=True)
class FrontierAudit:
    qualified_calls: int
    in_pool: tuple[tuple[int, int], ...]
    missing_ranked_method: tuple[str, ...]
    outside_ranked_owner: tuple[str, ...]
    ambiguous: tuple[str, ...]


def inspect_frontier(candidates: list[dict]) -> FrontierAudit:
    signatures = [parts(item["function_id"]) for item in candidates]
    simple_owners: dict[str, set[str]] = {}
    for owner, _, _ in signatures:
        short = owner.rsplit(".", 1)[-1]
        simple_owners.setdefault(short, set()).add(owner)
    edges: set[tuple[int, int]] = set()
    missing: set[str] = set()
    outside: set[str] = set()
    ambiguous: set[str] = set()
    qualified = 0
    for i, candidate in enumerate(candidates):
        for line in candidate["levels"][1].splitlines():
            if not line.strip().startswith("CALL "):
                continue
            call = _call_signature(line.strip()[5:])
            if call is None:
                continue
            receiver, method, arity = call
            # Only explicit uppercase simple-class receivers. Lowercase
            # receivers are often variables; full symbol identity requires
            # type attribution, and is deliberately not guessed here.
            if receiver is None or not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", receiver):
                continue
            qualified += 1
            possible_owners = simple_owners.get(receiver, set())
            label = f"{receiver}.{method}/{arity}"
            if len(possible_owners) > 1:
                ambiguous.add(label + ":owner_collision")
                continue
            if not possible_owners:
                outside.add(label)
                continue
            owner = next(iter(possible_owners))
            matches = [
                j for j, (candidate_owner, candidate_method, count) in enumerate(signatures)
                if i != j and candidate_owner == owner
                and candidate_method == method and count == arity
            ]
            if len(matches) == 1:
                edges.add((i, matches[0]))
            elif len(matches) > 1:
                ambiguous.add(label + ":overloaded_target")
            else:
                missing.add(label)
    return FrontierAudit(
        qualified_calls=qualified,
        in_pool=tuple(sorted(edges)),
        missing_ranked_method=tuple(sorted(missing)),
        outside_ranked_owner=tuple(sorted(outside)),
        ambiguous=tuple(sorted(ambiguous)),
    )
