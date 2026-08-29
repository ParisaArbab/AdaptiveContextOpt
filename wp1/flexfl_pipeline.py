"""FlexFL-aligned Agent4SR -> merge top-20 -> Agent4LR pipeline.

This module follows the original replication's two-stage design, but replaces
its pre-generated source corpus with a Graphify-backed view of the actual
Defects4J checkout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from evaluation import method_matches
from graphify_structure import GraphifyIndex
from llm_backends import ChatBackend


SR_SYSTEM = """You are Agent4SR, the space-reduction agent from a FlexFL-style fault-localization pipeline.
Your goal is to identify five suspicious Java methods for the failing Defects4J bug.
Use repository tools instead of guessing from names alone. At each step return EITHER one tool call OR the final ranking.
Allowed tool calls:
find_class(query)
find_method(query)
get_paths(query)
get_classes_of_path(path)
get_methods_of_class(class_name)
get_code_snippet_of_method(method_ref)

When ready, return exactly this format:
Top_1 : fully.qualified.Class.method(signature)
Top_2 : fully.qualified.Class.method(signature)
Top_3 : fully.qualified.Class.method(signature)
Top_4 : fully.qualified.Class.method(signature)
Top_5 : fully.qualified.Class.method(signature)
Do not output a patch. Do not output files only, the target is method-level localization."""


LR_SYSTEM = """You are Agent4LR, the local reranking agent from FlexFL.
You receive at most 20 candidate methods: top 5 from SBIR, top 5 from Ochiai,
top 5 from BoostN, and top 5 from Agent4SR. Rank the five most likely faulty methods.
You may inspect a candidate only with get_code_snippet_of_method(N), where N is its 1-based candidate number.
Your final answers MUST be chosen from the candidate list.

When ready, return exactly:
Top_1 : candidate method
Top_2 : candidate method
Top_3 : candidate method
Top_4 : candidate method
Top_5 : candidate method"""


@dataclass
class AgentRun:
    predictions: list[str]
    transcript: list[dict]
    final_response: str
    steps: int

    def to_dict(self) -> dict:
        return asdict(self)


def _base_evidence(bug: str, bug_report: str, trigger_test: str, runtime_output: str) -> str:
    return f"""BUG: {bug}

BUG REPORT:
{bug_report or '(not available)'}

TRIGGER TEST / FAILURE CONTEXT FROM FLEXFL:
{trigger_test or '(not available)'}

ACTUAL DEFECTS4J TEST OUTPUT FOR THIS EXPERIMENTAL ARM:
{runtime_output or '(empty output)'}
"""


def run_agent4sr(
    backend: ChatBackend,
    graph: GraphifyIndex,
    bug: str,
    bug_report: str,
    trigger_test: str,
    runtime_output: str,
    max_steps: int = 12,
) -> AgentRun:
    evidence = _base_evidence(bug, bug_report, trigger_test, runtime_output)
    transcript: list[dict] = []
    final_response = ""

    for step in range(1, max_steps + 1):
        user = evidence + _format_transcript(transcript) + "\nChoose the next tool call or give the final Top_1..Top_5 ranking."
        response = backend.complete(SR_SYSTEM, user)
        final_response = response
        predictions = parse_top5(response)
        if predictions:
            return AgentRun(predictions[:5], transcript, response, step)

        action = _parse_tool_call(response)
        if action is None:
            transcript.append({"assistant": response, "tool": "FORMAT_ERROR: use one allowed tool call or Top_1..Top_5"})
            continue
        name, arg = action
        result = _run_sr_tool(graph, name, arg)
        transcript.append({"assistant": response, "tool": result})

    return AgentRun(parse_top5(final_response)[:5], transcript, final_response, max_steps)


def run_agent4lr(
    backend: ChatBackend,
    graph: GraphifyIndex,
    bug: str,
    bug_report: str,
    trigger_test: str,
    runtime_output: str,
    candidates: list[str],
    max_steps: int = 12,
) -> AgentRun:
    evidence = _base_evidence(bug, bug_report, trigger_test, runtime_output)
    candidate_text = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, 1))
    transcript: list[dict] = []
    final_response = ""

    for step in range(1, max_steps + 1):
        user = (
            evidence
            + "\nMERGED CANDIDATES, original FlexFL order:\n"
            + candidate_text
            + _format_transcript(transcript)
            + "\nInspect another candidate or give the final Top_1..Top_5 ranking."
        )
        response = backend.complete(LR_SYSTEM, user)
        final_response = response
        raw_predictions = parse_top5(response)
        if raw_predictions:
            constrained = _constrain_to_candidates(raw_predictions, candidates)
            if constrained:
                return AgentRun(constrained[:5], transcript, response, step)

        action = _parse_tool_call(response)
        if action and action[0] == "get_code_snippet_of_method":
            try:
                index = int(action[1].strip())
            except ValueError:
                result = "ERROR: method number must be an integer from the candidate list"
            else:
                if 1 <= index <= len(candidates):
                    method = candidates[index - 1]
                    result = f"Candidate {index}: {method}\n{graph.snippet(method)}"
                else:
                    result = f"ERROR: candidate number must be 1..{len(candidates)}"
            transcript.append({"assistant": response, "tool": result})
        else:
            transcript.append({"assistant": response, "tool": "FORMAT_ERROR: use get_code_snippet_of_method(N) or Top_1..Top_5"})

    return AgentRun(_constrain_to_candidates(parse_top5(final_response), candidates)[:5], transcript, final_response, max_steps)


def parse_top5(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"(?im)^\s*Top[_\s-]?(\d+)\s*:\s*(.+?)\s*$", text or ""):
        rank = int(match.group(1))
        value = match.group(2).strip().strip("`* ")
        if 1 <= rank <= 5 and value:
            found.append((rank, value))
    found.sort(key=lambda x: x[0])
    out: list[str] = []
    for _, value in found:
        if value not in out:
            out.append(value)
    return out


def _parse_tool_call(text: str) -> tuple[str, str] | None:
    allowed = {
        "find_class",
        "find_method",
        "get_paths",
        "get_classes_of_path",
        "get_methods_of_class",
        "get_code_snippet_of_method",
    }
    for name, arg in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", text or "", flags=re.S):
        if name in allowed:
            return name, arg.strip().strip("'\"` ")
    return None


def _run_sr_tool(graph: GraphifyIndex, name: str, arg: str) -> str:
    if name == "find_class":
        return _list_result("classes", graph.find_classes(arg, 20))
    if name == "find_method":
        return _list_result("methods", graph.find_methods(arg, 30))
    if name == "get_paths":
        return _list_result("paths", graph.find_paths(arg, 30))
    if name == "get_classes_of_path":
        values: list[str] = []
        seen: set[str] = set()
        q = arg.lower()
        for node in graph.nodes:
            if q not in node.source_file.lower():
                continue
            cls = node.callable_class or _class_from_label(node.label)
            if cls and cls not in seen:
                seen.add(cls)
                values.append(cls)
        return _list_result("classes", values[:30])
    if name == "get_methods_of_class":
        q = arg.lower().replace("$", ".")
        values = [
            n.label
            for n in graph.nodes
            if (n.callable or "(" in n.label)
            and (q in n.label.lower().replace("$", ".") or q in str(n.callable_class or "").lower().replace("$", "."))
        ]
        return _list_result("methods", _unique(values)[:40])
    if name == "get_code_snippet_of_method":
        return graph.snippet(arg)
    return f"ERROR: unsupported tool {name}"


def _format_transcript(transcript: list[dict]) -> str:
    if not transcript:
        return ""
    rows = ["\n\nTOOL HISTORY:"]
    for item in transcript:
        rows.append(f"Assistant: {item['assistant']}\nTool result:\n{item['tool']}")
    return "\n\n".join(rows)


def _list_result(label: str, values: list[str]) -> str:
    if not values:
        return f"No matching {label}."
    return "\n".join(f"{i}. {value}" for i, value in enumerate(values, 1))


def _class_from_label(label: str) -> str:
    head = label.split("(", 1)[0]
    return head.rsplit(".", 1)[0] if "." in head else ""


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _constrain_to_candidates(predictions: list[str], candidates: list[str]) -> list[str]:
    out: list[str] = []
    used: set[int] = set()
    for pred in predictions:
        match_index = None
        for i, candidate in enumerate(candidates):
            if i in used:
                continue
            if pred.strip() == candidate.strip() or method_matches(pred, candidate):
                match_index = i
                break
        if match_index is not None:
            used.add(match_index)
            out.append(candidates[match_index])
    return out
