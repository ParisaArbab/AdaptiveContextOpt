"""Plan bounded requests for *missing* Java method definitions.

This is the active-evidence-acquisition hypothesis for AITECH. Requests are
created from class-qualified CALL hints in frozen buggy-source sketches.
They are NOT source definitions, verified Java targets, or bug predictions.

Ranking: references from methods named in failing evidence first; then
existing suspiciousness rank, then repeated callers. Do not use gold labels.
"""
from __future__ import annotations

from dataclasses import dataclass

from .java_call_audit import _call_signature, parts


@dataclass(frozen=True)
class RetrievalRequest:
    owner: str
    method: str
    arity: int
    evidence_callers: tuple[int, ...]
    failure_anchor_caller: bool
    priority: float
    binding_status: str = "unverified_receiver_name_and_arity"


def plan_missing_definitions(
    snapshot: dict, *, max_requests: int = 5,
) -> tuple[RetrievalRequest, ...]:
    if type(max_requests) is not int or max_requests < 0:
        raise ValueError("max_requests must be a nonnegative integer")
    candidates = snapshot["candidates"]
    signatures = [parts(item["function_id"]) for item in candidates]
    class_to_owners: dict[str, set[str]] = {}
    for owner, _, _ in signatures:
        class_to_owners.setdefault(owner.rsplit(".", 1)[-1], set()).add(owner)
    context = snapshot["context"]
    anchors = {
        i for i, (_, method, _) in enumerate(signatures)
        if method + "(" in context
    }
    missing: dict[tuple[str, str, int], set[int]] = {}
    for caller, candidate in enumerate(candidates):
        for line in candidate["levels"][1].splitlines():
            if not line.strip().startswith("CALL "):
                continue
            parsed = _call_signature(line.strip()[5:])
            if parsed is None:
                continue
            receiver, method, arity = parsed
            # A receiver that is not an uppercase type-like symbol may be
            # an instance variable. Avoid assigning it a class identity.
            if receiver is None or not receiver[0].isupper():
                continue
            owners = class_to_owners.get(receiver, set())
            if len(owners) != 1:
                continue
            owner = next(iter(owners))
            if any(o == owner and m == method and a == arity
                   for o, m, a in signatures):
                continue
            missing.setdefault((owner, method, arity), set()).add(caller)
    requests = []
    for (owner, method, arity), callers in missing.items():
        failure_call = bool(callers & anchors)
        # This is a transparent heuristic, not calibrated retrieval confidence.
        priority = (2.0 if failure_call else 0.0) + sum(
            1.0 / (1 + i) for i in callers
        )
        requests.append(RetrievalRequest(
            owner, method, arity, tuple(sorted(callers)),
            failure_call, round(priority, 6),
        ))
    return tuple(sorted(
        requests,
        key=lambda x: (-x.priority, x.owner, x.method, x.arity),
    )[:max_requests])
