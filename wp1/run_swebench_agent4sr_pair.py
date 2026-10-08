from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from wp1.graphify_structure import GraphifyIndex
from wp1.llm_backends import ChatBackend


SYSTEM = """You are Agent4SR for SWE-bench fault localization.


INVESTIGATION RULES:

1. Use Graphify tools to investigate before giving the final ranking.

2. Do not call the same tool with the same argument repeatedly.
   If you already inspected an entity or file, move to a new relevant
   entity instead of repeating the same lookup.

3. Follow structural relationships.
   If a suspicious class inherits from another class, parent, base class,
   trait, or mixin, inspect relevant unexplored parents before concluding.

4. When a symptom may be inherited from a parent class, do not assume the
   immediate child is faulty. Trace the inheritance chain far enough to
   identify where the behavior is introduced.

5. Prefer production-code entities over test functions in the final ranking.

6. Available investigation tools are only:
   find_path(...)
   find_function(...)
   get_functions_of_path(...)
   get_code_snippet(...)

   During investigation steps, output EXACTLY ONE tool call and nothing else.
   Do not explain your reasoning before or after the tool call.
   Do not simulate or write a Tool result.
   Do not invent shell commands such as grep, sed, cat, bash, or python.

7. Do not give a final Top_1..Top_5 ranking too early.
   Investigate first.

8. On the LAST step you MUST stop using tools and return exactly:

Top_1 : path/to/file.py::Entity
Top_2 : path/to/file.py::Entity
Top_3 : path/to/file.py::Entity
Top_4 : path/to/file.py::Entity
Top_5 : path/to/file.py::Entity


Your task is to identify the five most suspicious production-code entities
that may contain the bug.

A code entity may be:
- a function,
- a method,
- a class,
- or module-level code.

Every final candidate MUST include its source file.

You receive:
1. the SWE-bench problem statement,
2. the failing test name,
3. the runtime test output,
4. access to the buggy repository through Graphify.

Do NOT propose a patch.
Do NOT use the gold patch.
Do NOT rank test code unless the actual bug is in test code.

You MUST investigate the repository with Graphify before giving the final
ranking.

You MUST:
1. perform repository discovery,
2. inspect at least one source-code snippet,
3. trace relevant production-code classes and their inheritance,
4. then produce the final Top-5.

IMPORTANT SEARCH STRATEGY:
- The failing test is evidence, not usually the faulty production location.
- Do not spend many steps repeatedly inspecting test code.
- After understanding the failure, move quickly into production code.
- For inheritance-related bugs, inspect each parent class definition.
- If a class has correct __slots__, inspect its parent classes.
- Continue upward through the inheritance chain until you find the class
  responsible for the behavior.
- Prefer exact file::entity get_code_snippet calls once a source path is known.
- Do not repeat the same search or snippet request.

Available tools:

find_path(query)
find_function(query)
get_functions_of_path(path)
get_code_snippet(function)

IMPORTANT:
After you know both a file and entity, always use:

get_code_snippet("path/to/file.py::Entity")

For example:

get_code_snippet("sympy/core/symbol.py::Symbol")

Do not invent a file/entity combination.
If one lookup fails, change your search strategy instead of repeating it.

Examples of tool calls:

find_path("symbol")
find_function("Symbol")
get_functions_of_path("package/module.py")
get_code_snippet(".Symbol()")

Do not repeat the same unsuccessful query. Simplify or change the query.

FINAL OUTPUT FORMAT:

Top_1 : path/to/file.py::entity
Top_2 : path/to/file.py::entity
Top_3 : path/to/file.py::entity
Top_4 : path/to/file.py::entity
Top_5 : path/to/file.py::entity

Examples of valid entity formats:

package/module.py::ClassName
package/module.py::ClassName.method
package/module.py::function_name
package/module.py::<module>

Return exactly five candidates when finished.
"""


def parse_top5(text):
    found = []
    for m in re.finditer(
        r"(?im)^\s*Top[_\s-]?(\d+)\s*:\s*(.+?)\s*$",
        text or "",
    ):
        rank = int(m.group(1))
        value = m.group(2).strip().strip("`* ")
        if 1 <= rank <= 5 and value:
            found.append((rank, value))

    found.sort()
    result = []
    seen = set()
    for _, value in found:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) == 5:
            break
    return result


def _previously_seen_entities(final_response, history):
    """Collect path::entity strings already seen before format finalization."""
    values = []
    seen = set()
    texts = [final_response or ""]
    for item in history:
        texts.append(str(item.get("assistant", "")))
        texts.append(str(item.get("tool", "")))

    pattern = re.compile(
        r"(?<![\w./-])([A-Za-z0-9_./-]+\.py::[A-Za-z0-9_.$<>-]+)"
    )
    for text in texts:
        for match in pattern.finditer(text):
            value = match.group(1).rstrip(".,;:)")
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def finalize_top5_format(backend, final_response, history):
    """One format-only retry using no new tools, repository data, or gold."""
    seen_entities = _previously_seen_entities(final_response, history)
    candidates = "\\n".join(f"- {x}" for x in seen_entities[:40]) or "(none)"

    system = """You are a strict output formatter for an already completed
software fault-localization run.

You MUST NOT investigate, call tools, use gold information, or introduce new
repository evidence. Use only the previous conclusion and entities already seen
during the completed run.

Return exactly five lines and nothing else:
Top_1 : path/to/file.py::Entity
Top_2 : path/to/file.py::Entity
Top_3 : path/to/file.py::Entity
Top_4 : path/to/file.py::Entity
Top_5 : path/to/file.py::Entity
"""

    prompt = f"""PREVIOUS FINAL RESPONSE:
{final_response}

ENTITIES ALREADY SEEN DURING THIS RUN:
{candidates}

Rewrite the completed localization result into exactly five Top_1..Top_5 lines.
Preserve the previous ranking as much as possible. If the previous response was
truncated before Top_5, choose the missing candidate only from the already-seen
entities above. Do not add explanation.
"""

    print("[Agent4SR] FORMAT-ONLY FINALIZATION CALL", flush=True)
    response = backend.complete(system, prompt)
    print("[Agent4SR] FORMAT-ONLY RESPONSE:", flush=True)
    print(response, flush=True)
    predictions = parse_top5(response)
    return response, predictions


def _evidence_only_candidates(history, persistent_hypotheses):
    """Build unique finalization candidates without new repository access."""
    seen_entities = _previously_seen_entities("", history)
    allowed = []
    seen = set()

    def add(value):
        if value and value not in seen:
            seen.add(value)
            allowed.append(value)

    for value in persistent_hypotheses or []:
        add(value)
    for value in seen_entities:
        add(value)

    # Module-level entities are valid Agent4SR candidates and require no new
    # repository evidence. They are derived only from files already inspected.
    inspected_files = []
    for value in list(allowed):
        if "::" not in value:
            continue
        path = value.split("::", 1)[0]
        if path.endswith(".py") and path not in inspected_files:
            inspected_files.append(path)
    for path in inspected_files:
        add(f"{path}::<module>")

    return allowed


def _normalize_evidence_only_top5(predictions, allowed):
    """Return five unique allowed candidates without any new investigation."""
    allowed = [x for x in allowed if x]
    allowed_set = set(allowed)
    result = []
    seen = set()

    for value in predictions or []:
        if value not in allowed_set or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) == 5:
            return result

    for value in allowed:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) == 5:
            break

    return result


def finalize_slice_evidence_only(
    backend,
    *,
    evidence,
    history,
    persistent_hypotheses,
):
    """Finalize a slicing run without any additional repository investigation."""
    allowed = _evidence_only_candidates(history, persistent_hypotheses)
    candidates = "\n".join(f"- {x}" for x in allowed[:60]) or "(none)"
    history_text = ""
    for item in history:
        history_text += (
            f"\nAssistant:\n{item.get('assistant', '')}\n"
            f"Tool result:\n{item.get('tool', '')}\n"
        )

    system = """You are the evidence-only final ranking stage for software
fault localization.

No more tools are available. Do not investigate further. Do not use gold
information. Do not introduce repository entities that were not already seen.

Use only:
1. the failure evidence,
2. dependency-derived slicing hypotheses,
3. source/tool evidence already present in the transcript,
4. the allowed entity list.

Dependency-derived hypotheses are not ground truth. Keep them only when the
inspected evidence supports them.

Every ranked entity MUST be unique. Never repeat a candidate at two ranks.
Choose only from the allowed entity list below.

Return exactly five lines and nothing else:
Top_1 : path/to/file.py::Entity
Top_2 : path/to/file.py::Entity
Top_3 : path/to/file.py::Entity
Top_4 : path/to/file.py::Entity
Top_5 : path/to/file.py::Entity
"""

    prompt = f"""FAILURE AND SLICING EVIDENCE:
{evidence}

COMPLETED TOOL TRANSCRIPT:
{history_text}

ALLOWED ENTITIES ALREADY SEEN:
{candidates}

The investigation phase is finished. Rank the five most suspicious UNIQUE
production entities using only the evidence above. Every Top-N entry must be
different and must come from ALLOWED ENTITIES ALREADY SEEN. Do not call a tool.
Do not add any explanation.
"""

    print("[Agent4SR] SLICE EVIDENCE-ONLY FINALIZATION", flush=True)
    response = backend.complete(system, prompt)
    print("[Agent4SR] SLICE FINAL RESPONSE:", flush=True)
    print(response, flush=True)

    parsed = parse_top5(response)
    normalized = _normalize_evidence_only_top5(parsed, allowed)

    if len(normalized) == 5 and normalized != parsed:
        print(
            "[Agent4SR] NORMALIZED FINAL TOP-5: removed duplicates/invalid "
            "entries and filled only from already-seen evidence.",
            flush=True,
        )
        response = "\n".join(
            f"Top_{index} : {candidate}"
            for index, candidate in enumerate(normalized, 1)
        )

    return response, normalized


def parse_tool(text):
    """
    Parse Agent4SR tool calls.

    Accept both:
        find_path("sympy/core/symbol.py")

    and LLM-style keyword calls:
        find_path(query="sympy/core/symbol.py")
        get_functions_of_path(path="sympy/core/symbol.py")
        get_code_snippet(function=".Symbol()")
    """
    allowed = {
        "find_path",
        "find_function",
        "get_functions_of_path",
        "get_code_snippet",
    }

    pattern = (
        r"\b("
        + "|".join(sorted(allowed, key=len, reverse=True))
        + r")\s*\((.*?)\)"
    )

    match = re.search(pattern, text or "", flags=re.S)

    if not match:
        return None

    name = match.group(1)
    arg = match.group(2).strip()

    # Remove optional keyword syntax produced by LLMs:
    # query="...", path="...", function="...", method="..."
    kw = re.match(
        r"^(?:query|path|function|method|name|function_ref)\s*=\s*(.*)$",
        arg,
        flags=re.S,
    )

    if kw:
        arg = kw.group(1).strip()

    # Remove wrapping markdown/quotes.
    arg = arg.strip().strip("`").strip()

    if (
        len(arg) >= 2
        and arg[0] in {"'", '"'}
        and arg[-1] == arg[0]
    ):
        arg = arg[1:-1]

    return name, arg.strip()


def run_tool(graph, name, arg):
    if name == "find_path":
        values = graph.find_paths(arg, 30)
        return "\n".join(f"{i}. {x}" for i, x in enumerate(values, 1)) or "No paths found."

    if name == "find_function":
        values = graph.find_methods(arg, 40)
        return "\n".join(f"{i}. {x}" for i, x in enumerate(values, 1)) or "No functions found."

    if name == "get_functions_of_path":
        q = arg.lower()
        values = []
        seen = set()

        for node in graph.nodes:
            if q not in node.source_file.lower():
                continue
            if not node.callable and "(" not in node.label:
                continue
            if node.label in seen:
                continue

            seen.add(node.label)
            values.append(node.label)

        return "\n".join(
            f"{i}. {x}" for i, x in enumerate(values[:50], 1)
        ) or "No functions found."

    if name == "get_code_snippet":
        return graph.snippet(arg)

    return "Unsupported tool."


def run_agent(
    backend,
    graph,
    problem,
    failing_test,
    runtime_output,
    max_steps=20,
    targeted_evidence=None,
    previous_predictions=None,
    persistent_hypotheses=None,
    structural_focus=None,
    post_slice_extra_tool_budget=1,
):
    history = []

    evidence = f"""SWE-BENCH PROBLEM:

{problem}

FAILING TEST:
{failing_test}

RUNTIME TEST OUTPUT:

{runtime_output}
"""

    if targeted_evidence:
        evidence += "\nTARGETED FEEDBACK EVIDENCE:\n"
        evidence += (
            "The following evidence was retrieved because a previous feedback "
            "round identified a concrete unresolved question. It is real "
            "runtime/source evidence, not gold information. Use it together "
            "with Graphify and continue investigating as needed.\n"
        )
        for index, item in enumerate(targeted_evidence, 1):
            evidence += (
                f"\n[{index}] Evidence type: {item.get('evidence_type', '')}\n"
                f"Anchor: {item.get('anchor_entity', '')}\n"
                f"Question: {item.get('question', '')}\n"
                f"Evidence:\n{item.get('content', '')}\n"
            )

    persistent_hypotheses = list(persistent_hypotheses or [])
    if persistent_hypotheses:
        evidence += "\nDEPENDENCY-DERIVED PERSISTENT HYPOTHESES (NOT GROUND TRUTH):\n"
        evidence += (
            "These candidates were derived mechanically from the program slice "
            "and dependency graph, not from gold labels. Treat them as persistent "
            "hypotheses. Inspect them directly and keep them in consideration "
            "unless new source/runtime evidence contradicts them. Do not drift "
            "toward unrelated methods merely because more tool calls are available.\n"
        )
        for index, candidate in enumerate(persistent_hypotheses, 1):
            evidence += f"{index}. {candidate}\n"

    if structural_focus:
        evidence += "\nSTRUCTURAL FOCUS FOR THIS RUN:\n"
        evidence += str(structural_focus).strip() + "\n"

    previous_predictions = list(previous_predictions or [])
    if previous_predictions:
        evidence += "\nPREVIOUS LOCALIZATION STATE (NOT GROUND TRUTH):\n"
        evidence += (
            "These candidates came from the immediately previous Agent4SR run. "
            "Carry them forward as hypotheses so the new run does not forget "
            "useful localization state. Re-evaluate them against the newly "
            "retrieved evidence and Graphify. Do not blindly copy them, but do "
            "not discard all previous candidates merely because the search "
            "trajectory changed. A prior candidate should be dropped only when "
            "new runtime/source evidence makes it less plausible.\n"
        )
        for index, candidate in enumerate(previous_predictions, 1):
            evidence += f"{index}. {candidate}\n"

    final = ""
    tool_calls = 0
    tool_names_used = set()
    seen_tool_calls = set()
    inspected_hypotheses = set()
    regression_guard_used = False
    slice_ready_tool_calls = None
    evidence_only_finalizer_used = False

    for step in range(1, max_steps + 1):
        print(f"\\n[Agent4SR] STEP {step}/{max_steps}", flush=True)
        history_text = ""

        if history:
            history_text = "\n\nTOOL HISTORY:\n"
            for item in history:
                history_text += (
                    f"\nAssistant:\n{item['assistant']}\n"
                    f"Tool result:\n{item['tool']}\n"
                )

        slice_hypotheses_ready = (
            bool(persistent_hypotheses)
            and set(persistent_hypotheses).issubset(inspected_hypotheses)
            and tool_calls >= 2
            and "get_code_snippet" in tool_names_used
        )

        if slice_hypotheses_ready and slice_ready_tool_calls is None:
            slice_ready_tool_calls = tool_calls
            print(
                "[Agent4SR] SLICE HYPOTHESES INSPECTED: post-slice tool budget "
                f"starts now ({post_slice_extra_tool_budget} extra tool call(s)).",
                flush=True,
            )

        extra_tools_after_slice = (
            0
            if slice_ready_tool_calls is None
            else tool_calls - slice_ready_tool_calls
        )

        if (
            slice_hypotheses_ready
            and extra_tools_after_slice >= post_slice_extra_tool_budget
        ):
            final_response, final_predictions = finalize_slice_evidence_only(
                backend,
                evidence=evidence,
                history=history,
                persistent_hypotheses=persistent_hypotheses,
            )
            evidence_only_finalizer_used = True
            if len(final_predictions) != 5:
                print(
                    "[Agent4SR] SLICE FINALIZATION STOPPED: fewer than five "
                    "unique evidence-supported candidates were available. "
                    "No further Graphify calls will be made.",
                    flush=True,
                )
                return {
                    "predictions": final_predictions,
                    "steps": step - 1,
                    "tool_calls": tool_calls,
                    "tools_used": sorted(tool_names_used),
                    "final_response": final_response,
                    "previous_predictions_supplied": previous_predictions,
                    "persistent_hypotheses_supplied": persistent_hypotheses,
                    "inspected_persistent_hypotheses": sorted(inspected_hypotheses),
                    "regression_guard_used": regression_guard_used,
                    "evidence_only_finalizer_used": True,
                    "post_slice_extra_tool_budget": post_slice_extra_tool_budget,
                    "finalization_complete": False,
                    "transcript": history,
                }

            if len(final_predictions) == 5:
                return {
                    "predictions": final_predictions,
                    "steps": step - 1,
                    "tool_calls": tool_calls,
                    "tools_used": sorted(tool_names_used),
                    "final_response": final_response,
                    "previous_predictions_supplied": previous_predictions,
                    "persistent_hypotheses_supplied": persistent_hypotheses,
                    "inspected_persistent_hypotheses": sorted(inspected_hypotheses),
                    "regression_guard_used": regression_guard_used,
                    "evidence_only_finalizer_used": True,
                    "post_slice_extra_tool_budget": post_slice_extra_tool_budget,
                    "transcript": history,
                }

        if step == max_steps:
            step_instruction = (
                "\nTHIS IS THE FINAL STEP. "
                "DO NOT CALL ANY TOOL. "
                "Return exactly five ranked production-code entities now, "
                "using exactly this format:\n"
                "Top_1 : path/to/file.py::Entity\n"
                "Top_2 : path/to/file.py::Entity\n"
                "Top_3 : path/to/file.py::Entity\n"
                "Top_4 : path/to/file.py::Entity\n"
                "Top_5 : path/to/file.py::Entity"
            )
        elif slice_hypotheses_ready:
            remaining_extra = max(
                0,
                post_slice_extra_tool_budget - extra_tools_after_slice,
            )
            step_instruction = (
                "\nThe dependency-derived hypotheses have been directly "
                "inspected. The search is now bounded. "
                f"You have at most {remaining_extra} additional relevant "
                "Graphify tool call(s) before evidence-only final ranking. "
                "If one unresolved structural question remains, return EXACTLY "
                "ONE directly relevant Graphify tool call. Otherwise return a "
                "Top_1..Top_5 ranking now. Do not explore unrelated behavior."
            )
        else:
            step_instruction = (
                "\nReturn EXACTLY ONE new Graphify tool call and nothing else. "
                "No explanation. No markdown. No Tool result. "
                "Use only find_path(...), find_function(...), "
                "get_functions_of_path(...), or get_code_snippet(...)."
            )

        prompt = (
            evidence
            + history_text
            + step_instruction
            + (
                "\nIMPORTANT: This is the LAST available step. "
                "Stop searching and return your best Top_1..Top_5 ranking now."
                if step == max_steps
                else ""
            )
        )
        print("[Agent4SR] calling LLM...", flush=True)
        response = backend.complete(SYSTEM, prompt)
        print("[Agent4SR] LLM RESPONSE:", flush=True)
        print(response, flush=True)
        final = response

        predictions = parse_top5(response)

        if len(predictions) == 5:
            graphify_ready = (
                tool_calls >= 2
                and "get_code_snippet" in tool_names_used
            )

            # Slicing-specific regression guard. If dependency-derived
            # hypotheses were directly inspected but every one disappears from
            # the final Top-5, allow one corrective reconsideration. This does
            # not use gold labels and does not force a specific rank.
            dropped_all_supported_hypotheses = (
                bool(inspected_hypotheses)
                and not any(
                    candidate in predictions
                    for candidate in inspected_hypotheses
                )
            )

            if (
                graphify_ready
                and dropped_all_supported_hypotheses
                and not regression_guard_used
                and step < max_steps
            ):
                regression_guard_used = True
                print(
                    "[Agent4SR] SLICE REGRESSION GUARD: inspected dependency "
                    "hypotheses vanished from Top-5; requesting one reconsideration.",
                    flush=True,
                )
                history.append({
                    "assistant": response,
                    "tool": (
                        "SLICE REGRESSION GUARD. One or more dependency-derived "
                        "hypotheses were directly inspected earlier, but all were "
                        "dropped from the proposed Top-5. Reconsider the ranking "
                        "using the inspected source evidence. Keep a hypothesis "
                        "only if it remains plausible; if evidence contradicts it, "
                        "you may still exclude it. Return a revised Top-5 or one "
                        "new relevant Graphify tool call."
                    ),
                })
                continue

            if (
                graphify_ready
                and dropped_all_supported_hypotheses
                and not regression_guard_used
                and step == max_steps
            ):
                regression_guard_used = True
                print(
                    "[Agent4SR] FINAL SLICE REGRESSION GUARD: dependency "
                    "hypotheses vanished on the last step; running one "
                    "evidence-only reconsideration.",
                    flush=True,
                )
                guard_system = (
                    "You are performing a final evidence-only reconsideration "
                    "for software fault localization. Do not call tools, do not "
                    "use gold information, and do not invent repository facts. "
                    "Use only the dependency-derived hypotheses and source "
                    "evidence already inspected in the transcript. Return "
                    "exactly five Top_1..Top_5 lines."
                )
                guard_prompt = (
                    evidence
                    + history_text
                    + "\nPROPOSED FINAL RANKING THAT DROPPED ALL INSPECTED "
                    "SLICE HYPOTHESES:\n"
                    + response
                    + "\n\nReconsider whether the inspected dependency-derived "
                    "hypotheses remain plausible in light of the failure symptom "
                    "and their source snippets. They are not ground truth and "
                    "must not be forced into the ranking if contradicted. Return "
                    "a revised Top-5 using only already-seen entities."
                )
                guard_response = backend.complete(guard_system, guard_prompt)
                guard_predictions = parse_top5(guard_response)
                if len(guard_predictions) == 5:
                    predictions = guard_predictions
                    response = guard_response
                    final = guard_response

            if graphify_ready:
                return {
                    "predictions": predictions,
                    "steps": step,
                    "tool_calls": tool_calls,
                    "tools_used": sorted(tool_names_used),
                    "final_response": response,
                    "previous_predictions_supplied": previous_predictions,
                    "persistent_hypotheses_supplied": persistent_hypotheses,
                    "inspected_persistent_hypotheses": sorted(inspected_hypotheses),
                    "regression_guard_used": regression_guard_used,
                    "evidence_only_finalizer_used": evidence_only_finalizer_used,
                    "post_slice_extra_tool_budget": post_slice_extra_tool_budget,
                    "transcript": history,
                }

            print(
                "[Agent4SR] FINAL ANSWER REJECTED: "
                "Graphify investigation is required first.",
                flush=True,
            )

            history.append({
                "assistant": response,
                "tool": (
                    "You cannot give the final ranking yet. "
                    "Use Graphify for at least two tool calls and inspect "
                    "at least one source snippet with get_code_snippet()."
                ),
            })

            continue

        action = parse_tool(response)

        if not action:
            history.append({
                "assistant": "FORMAT_ERROR_RESPONSE_REJECTED",
                "assistant_raw": response,
                "tool": "FORMAT ERROR: use one allowed tool call or Top_1..Top_5",
            })
            continue

        name, arg = action

        # All Graphify tools in this runner accept exactly one string argument.
        # Reject accidental multi-argument calls instead of sending a malformed
        # path such as: get_code_snippet("file.py::Entity", 20, 30).
        if re.match(r"^['\"].*['\"]\s*,", arg):
            print(
                f"[Agent4SR] MULTI-ARG TOOL CALL REJECTED: {name}({arg})",
                flush=True,
            )
            history.append({
                "assistant": response,
                "tool": (
                    "INVALID TOOL CALL. This tool accepts exactly one string "
                    "argument. Retry with only the path/entity argument."
                ),
            })
            continue

        tool_key = (name, arg)

        if tool_key in seen_tool_calls:
            print(
                f"[Agent4SR] DUPLICATE TOOL CALL REJECTED: {name}({arg})",
                flush=True,
            )
            history.append({
                "assistant": f"{name}({json.dumps(arg)})",
                "assistant_raw": response,
                "tool": (
                    "DUPLICATE TOOL CALL REJECTED. "
                    "Choose a different unexplored Graphify query."
                ),
            })
            continue

        seen_tool_calls.add(tool_key)
        tool_calls += 1
        tool_names_used.add(name)

        print(f"[Agent4SR] TOOL CALL: {name}({arg})", flush=True)

        result = run_tool(graph, name, arg)

        if name == "get_code_snippet":
            for hypothesis in persistent_hypotheses:
                if arg.strip().strip("'\"") == hypothesis:
                    inspected_hypotheses.add(hypothesis)

        print("[Agent4SR] TOOL RESULT:", flush=True)
        print(result[:5000], flush=True)

        history.append({
            "assistant": f"{name}({json.dumps(arg)})",
            "assistant_raw": response,
            "tool": result,
        })

    predictions = parse_top5(final)
    format_finalization = None

    if len(predictions) != 5:
        formatted_response, formatted_predictions = finalize_top5_format(
            backend,
            final,
            history,
        )
        format_finalization = {
            "attempted": True,
            "original_prediction_count": len(predictions),
            "response": formatted_response,
            "prediction_count": len(formatted_predictions),
        }
        if len(formatted_predictions) == 5:
            predictions = formatted_predictions
            final = formatted_response

    return {
        "predictions": predictions,
        "steps": max_steps,
        "tool_calls": tool_calls,
        "tools_used": sorted(tool_names_used),
        "final_response": final,
        "format_finalization": format_finalization,
        "previous_predictions_supplied": previous_predictions,
        "persistent_hypotheses_supplied": persistent_hypotheses,
        "inspected_persistent_hypotheses": sorted(inspected_hypotheses),
        "regression_guard_used": regression_guard_used,
        "evidence_only_finalizer_used": evidence_only_finalizer_used,
        "post_slice_extra_tool_budget": post_slice_extra_tool_budget,
        "transcript": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", default="qwen3.6:27b")
    args = parser.parse_args()

    root = Path.home() / "AdaptiveContextOpt"
    work = root / "data/swebench_workspaces" / args.instance

    metadata = json.loads((work / "metadata.json").read_text())

    problem = metadata.get("problem_statement", "")
    failing = ", ".join(metadata.get("FAIL_TO_PASS", []))

    raw = (work / "outputs/raw_test_output.txt").read_text(errors="replace")
    lean = (work / "outputs/leanctx_test_output.txt").read_text(errors="replace")

    graph = GraphifyIndex.from_json(
        work / "repo",
        work / "repo/graphify-out/graph.json",
    )

    backend = ChatBackend(
        provider="ollama",
        model=args.model,
        timeout=1800,
    )

    print("\n===== RAW Agent4SR =====")
    raw_result = run_agent(
        backend,
        graph,
        problem,
        failing,
        raw,
    )

    print(json.dumps(raw_result["predictions"], indent=2))

    print("\n===== LeanCTX Agent4SR =====")
    lean_result = run_agent(
        backend,
        graph,
        problem,
        failing,
        lean,
    )

    print(json.dumps(lean_result["predictions"], indent=2))

    result = {
        "instance_id": args.instance,
        "model": args.model,
        "raw": raw_result,
        "leanctx": lean_result,
    }

    out = work / "outputs/agent4sr_pair.json"
    out.write_text(json.dumps(result, indent=2))

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
