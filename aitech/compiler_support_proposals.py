"""Convert compiler-attributed method calls into *unaudited* support proposals.

Call-graph adjacency is a source-grounded candidate for a debugging
explanation, not an independently sufficient explanation. All produced
proposals have sufficiency_audited=False; an explicit later audit
must justify promoting them to alternative support options.
"""
from __future__ import annotations

from dataclasses import asdict

from .java_compiler_graph import ResolvedCall
from .support_provenance import SupportProposal, proposal_summary
from .verified_graph_evidence import map_calls


def candidates_from_compiler(
    candidate_ids: list[str], calls: list[ResolvedCall]
) -> dict:
    mapped = map_calls(candidate_ids, calls)
    proposals = [
        SupportProposal(
            anchor=candidate_ids[caller],
            support=(candidate_ids[callee],),
            provenance="javac:static-call-target",
            compiler_binding_verified=True,
            sufficiency_audited=False,
        )
        for caller, callee in mapped.edges
    ]
    summary = proposal_summary(proposals, set(candidate_ids))
    return {
        "scope": "Compiler-attributed call graph to possible support, not fault causality",
        "compiler_edge_count": len(mapped.edges),
        "candidate_proposals": [asdict(p) for p in proposals],
        "sufficiency_gate": summary,
        "unmatched_target_count": len(mapped.unmatched_targets),
        "ambiguous_caller_count": len(mapped.ambiguous_callers),
        "ambiguous_target_count": len(mapped.ambiguous_targets),
        "ambiguous_short_type_identities": list(mapped.ambiguous_type_identities),
        "trained_dependency_detector": False,
        "automatic_sufficiency_promotion": False,
    }
