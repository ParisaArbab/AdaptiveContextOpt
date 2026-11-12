"""Provenance gate for sufficient debugging evidence paths.

Compiler symbol binding proves that a call resolves to a declaration;
it does not prove that call is a sufficient causal explanation of a
failed test. This module keeps proposed support links separate from
human/independently audited sufficient paths.

No automatic promotion from graph adjacency or source declaration to
causal sufficiency. Existing exact selector consumes only audited paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .alternative_support import SupportMap


@dataclass(frozen=True)
class SupportProposal:
    anchor: str
    support: tuple[str, ...]
    provenance: str
    compiler_binding_verified: bool
    sufficiency_audited: bool
    audit_reference: str = ""


def audited_support_sets(
    proposals: Iterable[SupportProposal], known_ids: set[str]
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Return OR-of-AND support options that passed independent sufficiency audit.

    A source-grounded edge alone is insufficient. Every retained support
    set must have an explicit audit reference and compiler-verified
    binding where the support claim relies on a Java call.
    """
    accepted: dict[str, set[tuple[str, ...]]] = {}
    for proposal in proposals:
        if not proposal.anchor or proposal.anchor not in known_ids:
            raise ValueError("unknown anchor")
        if not proposal.support or not set(proposal.support).issubset(known_ids):
            raise ValueError("unknown or empty support")
        if proposal.anchor in proposal.support:
            raise ValueError("self support cannot be sufficient")
        if not proposal.provenance:
            raise ValueError("missing provenance")
        if not proposal.sufficiency_audited:
            continue
        if not proposal.compiler_binding_verified:
            raise ValueError("audited sufficiency requires verified call binding")
        if not proposal.audit_reference.strip():
            raise ValueError("audited sufficiency requires an audit reference")
        accepted.setdefault(proposal.anchor, set()).add(tuple(proposal.support))
    return {k: tuple(sorted(v)) for k, v in sorted(accepted.items())}


def proposal_summary(
    proposals: Iterable[SupportProposal], known_ids: set[str]
) -> dict:
    items = list(proposals)
    verified = audited_support_sets(items, known_ids)
    return {
        "candidate_proposals": len(items),
        "audited_sufficient_bundles": sum(len(v) for v in verified.values()),
        "unaudited_or_unverified": sum(not p.sufficiency_audited for p in items),
        "promotion_policy": "no compiler edge becomes sufficient without explicit audit",
        "audited_support_sets": {k: [list(x) for x in v] for k, v in verified.items()},
    }
