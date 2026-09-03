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
    return [x[1] for x in found][:5]


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


def run_agent(backend, graph, problem, failing_test, runtime_output, max_steps=20):
    history = []

    evidence = f"""SWE-BENCH PROBLEM:

{problem}

FAILING TEST:
{failing_test}

RUNTIME TEST OUTPUT:

{runtime_output}
"""

    final = ""
    tool_calls = 0
    tool_names_used = set()

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

        prompt = (
            evidence
            + history_text
            + (
                "\nChoose ONE new investigation tool call."
                if step < max_steps
                else
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

        if predictions:
            graphify_ready = (
                tool_calls >= 2
                and "get_code_snippet" in tool_names_used
            )

            if graphify_ready:
                return {
                    "predictions": predictions,
                    "steps": step,
                    "tool_calls": tool_calls,
                    "tools_used": sorted(tool_names_used),
                    "final_response": response,
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
                "assistant": response,
                "tool": "FORMAT ERROR: use one allowed tool call or Top_1..Top_5",
            })
            continue

        name, arg = action

        tool_calls += 1
        tool_names_used.add(name)

        print(f"[Agent4SR] TOOL CALL: {name}({arg})", flush=True)

        result = run_tool(graph, name, arg)

        print("[Agent4SR] TOOL RESULT:", flush=True)
        print(result[:5000], flush=True)

        history.append({
            "assistant": response,
            "tool": result,
        })

    return {
        "predictions": parse_top5(final),
        "steps": max_steps,
        "tool_calls": tool_calls,
        "tools_used": sorted(tool_names_used),
        "final_response": final,
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
