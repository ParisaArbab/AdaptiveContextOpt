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
    ambiguous_type_identities: tuple[str, ...] = ()


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


def _canonical_type(value: str) -> str:
    """Compiler signature with erased generics but preserved package names."""
    return _strip_generics(value).replace("...", "[]").strip()


def _types_match(candidate_types: tuple[str, ...], compiler_types: tuple[str, ...]) -> bool:
    if len(candidate_types) != len(compiler_types):
        return False
    for candidate, actual in zip(candidate_types, compiler_types):
        named = _canonical_type(candidate)
        resolved = _canonical_type(actual)
        # The candidate's qualified package name is authoritative.
        # Only an unqualified candidate type may use short-name matching.
        if "." in named and named != resolved:
            return False
        if "." not in named and _simple_type(named) != _simple_type(resolved):
            return False
    return True


def map_calls(candidates: list[str], calls: Iterable[ResolvedCall]) -> MappedGraph:
    """Match full types when known; abstain on short-name collisions.

    This only establishes candidate identity if the available signatures
    uniquely identify it. A short-name match remains approximate and is
    never allowed to conflate two distinct compiler-qualified signatures.
    """
    candidate_signatures = []
    for function_id in candidates:
        prefix, suffix = function_id.split("(", 1)
        owner, method = prefix.rsplit(".", 1)
        types = tuple(_split_params(suffix.rsplit(")", 1)[0]))
        candidate_signatures.append((owner, method, types))

    calls = list(calls)
    edges: set[tuple[int, int]] = set()
    missing: set[str] = set()
    amb_caller: set[str] = set()
    amb_target: set[str] = set()
    ambiguous_type_ids: set[str] = set()
    mapped = 0

    # Track every compiler-qualified signature that can match a candidate.
    # Collisions mean a shortened Java name cannot distinguish its identity.
    seen_types: dict[int, set[tuple[str, ...]]] = {
        i: set() for i in range(len(candidates))
    }
    for call in calls:
        for owner, method, raw, is_caller in (
            (call.caller_owner, call.caller_method,
             call.caller_parameter_types, True),
            (call.target_owner, call.target_method,
             call.parameter_types, False),
        ):
            if raw is None:
                continue
            compiler_types = tuple(_split_params(raw))
            compiler_canonical = tuple(map(_canonical_type, compiler_types))
            for i, (candidate_owner, candidate_method, candidate_types) in enumerate(candidate_signatures):
                if (owner == candidate_owner and method == candidate_method
                        and _types_match(candidate_types, compiler_types)):
                    seen_types[i].add(compiler_canonical)
    conflicting = {i for i, values in seen_types.items() if len(values) > 1}

    for call in calls:
        call_label = f"{call.caller_owner}.{call.caller_method}"
        caller_types = (tuple(_split_params(call.caller_parameter_types))
                        if call.caller_parameter_types is not None else None)
        callers = [
            i for i, (owner, name, types) in enumerate(candidate_signatures)
            if owner == call.caller_owner and name == call.caller_method
            and (caller_types is None or _types_match(types, caller_types))
        ]
        if len(callers) != 1:
            amb_caller.add(call_label)
            continue
        caller = callers[0]
        if caller in conflicting:
            ambiguous_type_ids.add(candidates[caller])
            continue

        callee_types = tuple(_split_params(call.parameter_types))
        targets = [
            i for i, (owner, name, types) in enumerate(candidate_signatures)
            if owner == call.target_owner and name == call.target_method
            and _types_match(types, callee_types)
        ]
        target_label = f"{call.target_owner}.{call.target_method}({call.parameter_types})"
        if not targets:
            missing.add(target_label)
        elif len(targets) > 1:
            amb_target.add(target_label)
        elif targets[0] in conflicting:
            ambiguous_type_ids.add(candidates[targets[0]])
        else:
            edges.add((caller, targets[0]))
            mapped += 1
    return MappedGraph(
        tuple(sorted(edges)), tuple(sorted(missing)),
        tuple(sorted(amb_caller)), tuple(sorted(amb_target)), mapped,
        tuple(sorted(ambiguous_type_ids))
    )
