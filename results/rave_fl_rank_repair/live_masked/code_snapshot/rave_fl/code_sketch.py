"""Local Java AST sketches. No LLM, patch, labels, or fixed bug-name rules.

L1 is a lossy representation, not an equivalence-preserving program slice.
Exact expressions retain operators, literals, exception types and dependencies.
Parsing failures are exposed as a feature and force full-code escalation.
"""
from __future__ import annotations

import re
from rave_fl.core import approx_tokens
from rave_fl.pilot_types import PilotCandidate


FEATURE_NAMES = (
    "prior", "rank_fraction", "ranking_gap", "ranker_agreement", "ranker_disagreement",
    "failure_relevance", "trace_match", "complexity", "call_count", "literal_count",
    "source_tokens", "l0_tokens", "l1_tokens", "compression_ratio", "parse_error",
    "sketch_truncated", "level", "suspicion", "uncertainty", "valid_feedback",
    "remaining_fraction", "inspected_fraction",
)


def words(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", text) if len(w) > 2}


def java_sketch(function_id: str, source: str, max_chars: int = 2200):
    from tree_sitter import Language, Parser
    import tree_sitter_java

    if not source.strip() or max_chars < 100:
        raise ValueError("A real nonempty method and a sketch limit >= 100 are required")
    parser = Parser(Language(tree_sitter_java.language()))
    prefix = b"class __RaveWrapper {\n"
    data = prefix + source.encode("utf-8") + b"\n}"
    root = parser.parse(data).root_node
    stack, nodes = [root], []
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(reversed(n.named_children))
    methods = [n for n in nodes if n.type in {"method_declaration", "constructor_declaration"}]
    method = methods[0] if methods else None
    body = method.child_by_field_name("body") if method else None
    signature = (data[method.start_byte:body.start_byte].decode().strip()
                 if method and body else function_id)
    l0 = f"METHOD: {function_id}\nSIGNATURE: {signature}"
    selected, seen = [], set()
    control = {"if_statement", "while_statement", "do_statement", "for_statement",
               "enhanced_for_statement", "switch_expression", "switch_statement", "catch_clause"}
    expression_types = {"return_statement", "throw_statement", "local_variable_declaration",
                        "assignment_expression", "update_expression"}
    call_count = literal_count = complexity = 0
    for n in nodes:
        # Work only inside the selected method, excluding wrapper and unrelated methods.
        if body is None or n.start_byte < body.start_byte or n.end_byte > body.end_byte:
            continue
        if n.type in control:
            complexity += 1
            head_end = next((c.start_byte for c in n.named_children
                             if c.type in {"block", "expression_statement", "return_statement", "throw_statement", "if_statement"}), n.end_byte)
            text = data[n.start_byte:head_end].decode().strip()
        elif n.type in expression_types:
            text = data[n.start_byte:n.end_byte].decode().strip()
        elif n.type == "method_invocation":
            call_count += 1
            text = "CALL " + data[n.start_byte:n.end_byte].decode().strip()
        else:
            if "literal" in n.type:
                literal_count += 1
            continue
        text = re.sub(r"\s+", " ", text)
        if text and text not in seen:
            seen.add(text)
            selected.append(text)
    structural = "\n".join(selected)
    full_sketch = l0 + "\nSTRUCTURE (lossy):\n" + structural
    truncated = len(full_sketch) > max_chars
    if truncated:
        full_sketch = full_sketch[:max_chars] + "\n[SKETCH TRUNCATED: unresolved code requires expansion or full verification]"
    # CSV spans may contain an unfinished Javadoc for the NEXT method. Only
    # errors inside this method should affect its risk, not wrapper trivia.
    metadata = {"parser": "tree-sitter-java", "parse_error": body is None or (method.has_error if method else True),
                "truncated": truncated, "complexity": complexity + 1,
                "call_count": call_count, "literal_count": literal_count}
    return (l0, full_sketch, source), metadata


def prepare_candidates(candidates, sources, metadata, context):
    prepared = []
    context_words = words(context)
    count = len(candidates)
    for i, c in enumerate(candidates):
        source = sources.get(c.function_id, "")
        levels, sketch_meta = java_sketch(c.function_id, source)
        item = metadata.get(c.function_id, {})
        ranks = list(item.get("ranks", {}).values())
        agreement = len(item.get("methods", [])) / 3
        disagreement = ((max(ranks) - min(ranks)) / max(1, item.get("top_per_method", 10))
                        if len(ranks) > 1 else 1 - agreement)
        # Lexical runtime relevance is a feature, never a mandatory selection rule.
        source_words = words(c.function_id + "\n" + source)
        relevance = len(context_words & source_words) / max(1, len(source_words))
        leaf = c.function_id.split("(")[0].rsplit(".", 1)[-1]
        f = {
            "prior": c.semantic_score,  # explicitly a rank proxy, not semantic similarity
            "rank_fraction": (i + 1) / max(1, count),
            "ranking_gap": max(0, c.semantic_score - (candidates[i + 1].semantic_score if i + 1 < count else 0)),
            "ranker_agreement": agreement, "ranker_disagreement": disagreement,
            "failure_relevance": relevance,
            "trace_match": float(bool(re.search(r"\b" + re.escape(leaf) + r"\b", context))),
            "complexity": float(sketch_meta["complexity"]),
            "call_count": float(sketch_meta["call_count"]),
            "literal_count": float(sketch_meta["literal_count"]),
            "source_tokens": float(approx_tokens(source)),
            "l0_tokens": float(approx_tokens(levels[0])),
            "l1_tokens": float(approx_tokens(levels[1])),
            "compression_ratio": min(1, approx_tokens(levels[1]) / max(1, approx_tokens(source))),
            "parse_error": float(sketch_meta["parse_error"]),
            "sketch_truncated": float(sketch_meta["truncated"]),
        }
        prepared.append(PilotCandidate(c.function_id, source, c.semantic_score, i + 1,
                                       levels, f, sketch_meta))
    if len({c.function_id for c in prepared}) != count:
        raise ValueError("Candidate IDs must be unique")
    return prepared


def state_features(candidate, status, remaining_fraction, inspected_fraction):
    obs = status.observation
    values = {**candidate.features, "level": float(status.level),
              "suspicion": obs.suspicion if obs else 0.5,
              "uncertainty": obs.uncertainty if obs else 1.0,
              "valid_feedback": float(obs is not None and obs.valid),
              "remaining_fraction": remaining_fraction, "inspected_fraction": inspected_fraction}
    return {key: float(values[key]) for key in FEATURE_NAMES}
