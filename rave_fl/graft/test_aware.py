"""Opt-in, test-aware, gold-free evidence selection and conservative call gating.

Only supplied failure text and code from the immutable buggy-revision graph are
used. In particular, 'fail()' after a call is a hint, not proof of what exception
is expected. Unknown/partial test source remains explicitly uncertain.
"""
from __future__ import annotations

from dataclasses import replace
import re

from rave_fl.graft.focused import call_names, resolve_callees, source_owners
from rave_fl.graft.selection import ByteCounter, mandatory_ids, package

SELECTOR = "test_aware_compact_v1"
POLICY = "l1_graph_test_aware"
_NO_MISSING = frozenset({"", "none", "n/a", "no", "not needed", "unknown"})


def failure_signal(text: str, evidence_origin: str = "") -> dict:
    """Conservative lexical reading of supplied failure/test excerpts only."""
    text = text or ""
    has_try = bool(re.search(r"\btry\s*\{", text))
    has_fail = bool(re.search(r"\bfail\s*\(\s*\)\s*;", text))
    catch = re.search(r"\bcatch\s*\(\s*([\w.$]+)\s+\w+", text)
    if has_try and has_fail:
        kind = "possible_expected_exception"
        certainty = "catch_present" if catch else "incomplete_test_excerpt"
    elif re.search(r"\bassert(?:Equals|True|False|Null|NotNull|Same|That)\s*\(", text):
        kind, certainty = "assertion", "observed_assertion_call"
    elif "AssertionFailedError" in text or "AssertionError" in text:
        kind, certainty = "assertion_failure", "trace_only"
    else:
        kind, certainty = "unknown", "insufficient_failure_context"
    return {
        "kind": kind,
        "certainty": certainty,
        "catch_type": catch.group(1) if catch else None,
        "evidence_origin": evidence_origin,
        "full_test_verified": False,
    }


def failure_source(graph) -> tuple[str, str]:
    failures = [n for n in graph.nodes if n.kind == "failure"]
    return (
        "\n".join(n.text for n in failures),
        failures[0].origin.extractor if failures else "",
    )


def test_signal(graph) -> dict:
    body, provenance = failure_source(graph)
    return failure_signal(body, provenance)


def actionable_missing(missing: str) -> bool:
    text = (missing or "").strip().lower().rstrip(". ")
    if text in _NO_MISSING:
        return False
    # Missing statements like "no evidence" identify a gap but not a
    # verifiable method reference; don't treat them as no remaining question.
    return True


def stop_decision(observation, rank: int, *, top_k: int = 5) -> str | None:
    """Opt-in, conservative early-stop criterion based on online L1 only.

    It never consults gold ranks, changes baseline policies, or claims that a
    high confidence score is calibrated probability. Stops only a top-1,
    high-suspicion, low-uncertainty candidate with NO open evidence request.
    Other candidates can still receive graph inspection.
    """
    if (
        observation is not None
        and observation.valid
        and rank == 1
        and top_k >= 1
        and observation.suspicion >= 0.85
        and observation.uncertainty <= 0.20
        and not actionable_missing(observation.missing_evidence)
    ):
        return "high_suspicion_no_open_evidence_question"
    return None


def _branch_score(node, signal, test_text, calls):
    """A transparent relevance heuristic, not dynamic reachability."""
    text = node.text
    header = text.strip().splitlines()[0] if text.strip() else ""
    score = 0
    if signal["kind"] == "possible_expected_exception":
        score += 12 if re.search(r"\b(?:throw|validate|checkArgument|checkState)\b", text) else 0
        score += 7 if re.search(r"\b(?:set|verify|assert)\s*\(", text) else 0
        score += 3 if re.match(r"(?:if|switch|try)\b", header) else 0
        score += 2 if re.search(r"\breturn\b", text) else 0
    elif signal["kind"] in {"assertion", "assertion_failure"}:
        score += 8 if re.search(r"\breturn\b", text) else 0
        score += 6 if re.search(r"\b(?:set|get|compare)\s*\(", text) else 0
    if any(re.search(r"\b" + re.escape(c) + r"\s*\(", text) for c in calls):
        score += 3
    if re.search(r"\b(?:if|switch|while|for)\s*\(", header):
        score += 1
    # Small nodes only break ties if semantic cues were equally strong;
    # avoid selecting arbitrary large source blocks merely for being long.
    return (score, -len(text), -node.origin.start_line, node.id)


def test_aware_select(graph, target, budget, counter=None, missing_evidence=""):
    """Return a bounded, test-conditioned source package, without gold data.

    Test signal has priority over unverified L1 missing-evidence suggestions.
    For ambiguous/missing callees, keep the target code and note the limitation
    instead of inventing a call relation or declaring semantic completeness.
    """
    counter = counter or ByteCounter()
    text, provenance = failure_source(graph)
    signal = failure_signal(text, provenance)
    requested = call_names(missing_evidence)
    resolved, unresolved = resolve_callees(graph, target, requested)
    owners = source_owners(graph)
    nodes = sorted(
        (n for n in owners.get(target, ()) if n.kind == "code"),
        key=lambda n: (n.origin.start_line, n.id),
    )
    chosen = set(mandatory_ids(graph, target))
    def try_add(more):
        proposed = chosen | {n.id for n in more}
        out = package(graph, target, proposed, budget, counter, selector=SELECTOR)
        if out.status == "ready":
            chosen.update(n.id for n in more)
            return True
        return False

    minimum = package(graph, target, chosen, budget, counter, selector=SELECTOR)
    if minimum.status != "ready":
        return replace(
            minimum,
            reason="test_aware_mandatory_exceeds_budget",
            diagnostics={"test_signal": signal, "required_missing": ["failure_and_target"]},
        )

    # Include the source units most relevant to exception/assertion behavior.
    # Each unit is whole AST text from the validated buggy checkout.
    used = []
    for node in sorted(nodes, key=lambda n: _branch_score(n, signal, text, requested), reverse=True):
        if try_add([node]):
            used.append(node.id)

    # Include a named callee only when its complete available source fits.
    # Otherwise keep the more relevant target behavior and report the gap.
    selected_callees = []
    for callee in resolved:
        if try_add(owners[callee]):
            selected_callees.append(callee)

    result = package(graph, target, chosen, budget, counter, selector=SELECTOR)
    missing_ids = [n.id for n in nodes if n.id not in chosen]
    diagnostics = {
        "test_signal": signal,
        "requested_calls": requested,
        "resolved_callees": resolved,
        "selected_callees": selected_callees,
        "unresolved_callees": unresolved,
        "missing_callee_bodies": [c for c in resolved if c not in selected_callees],
        "omitted_target_code_ids": missing_ids,
        "semantic_completeness_guaranteed": not missing_ids and not unresolved
            and len(selected_callees) == len(resolved),
        "test_aware_heuristic_calibrated": False,
        "runtime_reachability_proven": False,
        "gold_used": False,
    }
    return replace(result, diagnostics=diagnostics)


# Pytest must not collect this imported selector as a standalone test case.
test_aware_select.__test__ = False
