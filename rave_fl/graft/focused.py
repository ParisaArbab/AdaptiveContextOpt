"""Opt-in, gold-free focused retrieval from immutable buggy-revision graph nodes.

This is a *source-evidence* selector, not a runtime call-graph oracle.
Names in L1 feedback only guide retrieval when their calls are verified in
the actual target source and resolve to a unique known candidate method.
"""
from __future__ import annotations

from dataclasses import replace
import re

from rave_fl.graft.selection import ByteCounter, mandatory_ids, package

FOCUSED_SELECTOR = "focused_compact_v1"
_NAME = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
_IGNORED = frozenset({"if", "for", "while", "switch", "catch", "none", "null"})


def call_names(missing_evidence: str) -> list[str]:
    """Return real method-call hints, not guessed names from ordinary prose."""
    return list(dict.fromkeys(
        name for name in _NAME.findall(missing_evidence or "")
        if name.lower() not in _IGNORED
    ))


def source_owners(graph):
    """Map method owner to exact graph-backed signature and code nodes."""
    mapping = {}
    for node in graph.nodes:
        if node.kind in {"method", "code"} and node.owner:
            mapping.setdefault(node.owner, []).append(node)
    return mapping


def resolve_callees(graph, target: str, hints: list[str]):
    """Lexical fallback only for a *unique* source-supported method.

    MAY_CALL edges are not invented, and unknown/ambiguous callees are
    reported rather than being treated as proven runtime dependencies.
    """
    source = graph.sources[target]["source"]
    owners = source_owners(graph)
    resolved, unresolved = [], []
    for hint in hints:
        pattern = r"\b" + re.escape(hint) + r"\s*\("
        if not re.search(pattern, source):
            unresolved.append({"name": hint, "reason": "not_invoked_in_target_source"})
            continue
        matches = [
            owner for owner, nodes in owners.items()
            if owner != target
            and owner.rsplit(".", 1)[-1].split("(", 1)[0] == hint
            and any(n.kind == "method" for n in nodes)
            and any(n.kind == "code" for n in nodes)
        ]
        if len(matches) != 1:
            unresolved.append({
                "name": hint,
                "reason": "ambiguous_or_missing_source_method",
                "matching_method_count": len(matches),
            })
            continue
        resolved.append(matches[0])
    return list(dict.fromkeys(resolved)), unresolved


def focus_select(graph, target, budget, counter=None, missing_evidence=""):
    """Pack target flow + source-verified requested callee bodies first.

    Requires complete available code fragments for explicitly requested
    callees and the target branch using their returned variables. If a
    required bundle will not fit, returns *infeasible* rather than sending
    the graph verdict an incomplete or misleading package. Other target
    code remains optional and is listed as omitted.
    """
    counter = counter or ByteCounter()
    required = set(mandatory_ids(graph, target))
    owner_nodes = source_owners(graph)
    target_source = graph.sources[target]["source"]
    target_codes = sorted(
        (n for n in owner_nodes.get(target, ()) if n.kind == "code"),
        key=lambda n: (n.origin.start_line, n.id),
    )
    requested = call_names(missing_evidence)
    callees, unresolved = resolve_callees(graph, target, requested)
    selector = FOCUSED_SELECTOR

    def attempt(ids):
        return package(graph, target, ids, budget, counter, selector=selector)

    current = attempt(required)
    if current.status != "ready":
        return replace(
            current, reason="focused_mandatory_exceeds_budget",
            diagnostics={"requested_calls": requested, "required_missing": ["mandatory_failure_and_target"]},
        )

    chosen = set(required)
    required_missing = []
    selected_callees = []

    def add_bundle(nodes, label, essential=True):
        nonlocal chosen
        proposed = chosen | {n.id for n in nodes}
        trial = attempt(proposed)
        if trial.status == "ready":
            chosen = proposed
            return True
        if essential:
            required_missing.append(label)
        return False

    # Include the actual source-backed invocation for each requested method.
    hinted_target_calls = [
        n for n in target_codes
        if any(re.search(r"\b" + re.escape(h) + r"\s*\(", n.text) for h in requested)
    ]
    for node in hinted_target_calls:
        add_bundle([node], "target_invocation:" + str(node.origin.start_line))

    # The method signature plus *every* available source block of a named
    # callee are one atomic bundle. Avoid a signature-only "inspection".
    for callee in callees:
        nodes = owner_nodes[callee]
        if add_bundle(nodes, "callee_body:" + callee):
            selected_callees.append(callee)

    # Track variables assigned from a specifically named callee call.
    variables = []
    for name in requested:
        pattern = r"\b([A-Za-z_$][\w$]*)\s*=\s*(?:[A-Za-z_$][\w$]*\.)?" + re.escape(name) + r"\s*\("
        variables.extend(re.findall(pattern, target_source))
    variables = list(dict.fromkeys(variables))

    # Only branch headers, not arbitrary long blocks, get priority.
    def related_branch(node):
        first = node.text.strip().splitlines()[0] if node.text.strip() else ""
        control = re.match(r"(?:if|switch|while|for)\s*\(", first)
        return bool(control and any(re.search(r"\b" + re.escape(v) + r"\b", first) for v in variables))

    core = [n for n in target_codes if related_branch(n)]
    for node in core:
        add_bundle([node], "target_control_flow:" + str(node.origin.start_line))

    # Fill with remaining on-target code. Do not prioritize other unrelated
    # methods after the targeted call and target control-flow are supplied.
    optional = sorted(
        (n for n in target_codes if n.id not in chosen),
        key=lambda n: (
            0 if n.text.strip().startswith("if") else 1,
            n.origin.start_line, n.id,
        ),
    )
    for node in optional:
        add_bundle([node], "optional", essential=False)

    final = attempt(chosen)
    omitted_target = [n.id for n in target_codes if n.id not in chosen]
    diagnostics = {
        "requested_calls": requested,
        "resolved_callees": callees,
        "selected_callees": selected_callees,
        "unresolved_callees": unresolved,
        "assigned_variables": variables,
        "required_missing": required_missing,
        "omitted_target_code_ids": omitted_target,
        "semantic_completeness_guaranteed": not omitted_target and not unresolved and not required_missing,
        "gold_used": False,
        "heuristic_not_calibrated": True,
        "relations_are_lexical_not_proven_execution": True,
    }
    # If an explicit requested call could not be resolved, do not pretend to
    # have verified its body. Fail closed for this focused policy.
    if required_missing or unresolved:
        return replace(final, status="infeasible", reason="focused_evidence_incomplete", diagnostics=diagnostics)
    return replace(final, diagnostics=diagnostics)
