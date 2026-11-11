"""Map compiler-attributed static call targets to candidate evidence IDs.

Javac resolves declaration owners, method names and parameter types.
Archived ranking candidates may use short type names. Matching is
conservative: ambiguous overloads and unknown candidates are rejected.

A resolved compiler target is a reliable static declaration. A mapped
candidate is still an approximate evidence identity if method signatures
were normalized without the original source binding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .java_compiler_graph import ResolvedCall


@dataclass(frozen=True)
class MappedGraph:
    edges: tuple[tuple[int, int], ...]
    unmatched_targets: tuple[str, ...]
    ambiguous_callers: tuple[str, ...]
    ambiguous_targets: tuple[str, ...]
    mapped_calls: int


def _strip_generics(value: str) -> str:
    result = []
    depth = 0
    for char in value:
        if char == "<":
            depth += 1
        elif char == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            result.append(char)
    return "".join(result)


def _simple_type(type_name: str) -> str:
    type_name = _strip_generics(type_name).replace("...", "[]").strip()
    suffix = ""
    while type_name.endswith("[]"):
        suffix += "[]"
        type_name = type_name[:-2]
    return type_name.rsplit(".", 1)[-1] + suffix


def _split_params(signature: str) -> list[str]:
    if not signature.strip():
        return []
    parts, item, level = [], "", 0
    for char in signature:
        if char == "<":
            level += 1
        elif char == ">":
            level = max(0, level - 1)
        if char == "," and level == 0:
            parts.append(item.strip())
            item = ""
        else:
            item += char
    parts.append(item.strip())
    return parts


def _candidate_signature(function_id: str) -> tuple[str, str, tuple[str, ...]]:
    prefix, suffix = function_id.split("(", 1)
    owner, name = prefix.rsplit(".", 1)
    parameters = suffix.rsplit(")", 1)[0]
    return owner, name, tuple(map(_simple_type, _split_params(parameters)))


def map_calls(candidates: list[str], calls: Iterable[ResolvedCall]) -> MappedGraph:
    signatures = [_candidate_signature(identifier) for identifier in candidates]
    edges: set[tuple[int, int]] = set()
    missing: set[str] = set()
    amb_caller: set[str] = set()
    amb_target: set[str] = set()
    mapped = 0
    for call in calls:
        # Prefer compiler-attributed caller parameter types, not just
        # caller method names. The old 8-column format has no caller
        # signature, so it can map only a unique caller name.
        caller_params = (
            tuple(map(_simple_type, _split_params(call.caller_parameter_types)))
            if call.caller_parameter_types is not None else None
        )
        callers = [
            i for i, (owner, name, params) in enumerate(signatures)
            if owner == call.caller_owner and name == call.caller_method
            and (caller_params is None or params == caller_params)
        ]
        call_label = f"{call.caller_owner}.{call.caller_method}"
        if len(callers) != 1:
            amb_caller.add(call_label)
            continue
        desired = (
            call.target_owner,
            call.target_method,
            tuple(map(_simple_type, _split_params(call.parameter_types))),
        )
        targets = [i for i, signature in enumerate(signatures)
                   if signature == desired]
        target_label = f"{call.target_owner}.{call.target_method}({call.parameter_types})"
        if not targets:
            missing.add(target_label)
        elif len(targets) > 1:
            amb_target.add(target_label)
        else:
            edges.add((callers[0], targets[0]))
            mapped += 1
    return MappedGraph(
        tuple(sorted(edges)), tuple(sorted(missing)),
        tuple(sorted(amb_caller)), tuple(sorted(amb_target)), mapped
    )
