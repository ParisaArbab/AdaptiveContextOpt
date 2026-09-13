from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from wp1.graphify_structure import GraphifyIndex
from wp1.llm_backends import ChatBackend


SYSTEM = """You are Agent4LR, the local reranking agent in a FlexFL-style
fault-localization pipeline adapted to SWE-bench/Python.

You receive exactly 20 candidate production-code entities produced from:

- SBIR Top-5
- Ochiai Top-5
- BoostN Top-5
- Agent4SR Top-5

Your task is to rerank these candidates and return the five most likely
faulty entities.

IMPORTANT RULES:

1. Your final predictions MUST come from the provided candidate list.

2. You may inspect source code only through:

   get_code_snippet_of_candidate(N)

   where N is the 1-based candidate number.

3. Do not search for new candidates outside this list.

4. Do not propose a patch.

5. Do not use or request the gold patch.

6. Use the problem statement, failing test, runtime output, candidate list,
   and inspected snippets to decide which candidates are most suspicious.

7. When ready, return exactly:

Top_1 : path/to/file.py::Entity
Top_2 : path/to/file.py::Entity
Top_3 : path/to/file.py::Entity
Top_4 : path/to/file.py::Entity
Top_5 : path/to/file.py::Entity

At each step, return either ONE tool call or the final Top-5.
"""


def parse_top5(text: str) -> list[str]:
    found = []

    for match in re.finditer(
        r"(?im)^\s*Top[_\s-]?(\d+)\s*:\s*(.+?)\s*$",
        text or "",
    ):
        rank = int(match.group(1))
        value = match.group(2).strip().strip("`* ")

        if 1 <= rank <= 5 and value:
            found.append((rank, value))

    found.sort(key=lambda x: x[0])

    result = []

    for _, value in found:
        if value not in result:
            result.append(value)

    return result[:5]


def parse_tool(text: str):
    pattern = (
        r"\bget_code_snippet_of_candidate"
        r"\s*\(\s*(?:N\s*=\s*)?([0-9]+)\s*\)"
    )

    match = re.search(
        pattern,
        text or "",
        flags=re.I,
    )

    if not match:
        return None

    return int(match.group(1))


def normalize_prediction(
    prediction: str,
    candidates: list[str],
):
    value = prediction.strip().strip("`* ")

    # Exact entity match.
    for candidate in candidates:
        if value == candidate:
            return candidate

    # Allow "Candidate 7: entity".
    match = re.match(
        r"(?i)^candidate\s+(\d+)\s*:?\s*(.*)$",
        value,
    )

    if match:
        index = int(match.group(1))

        if 1 <= index <= len(candidates):
            remainder = match.group(2).strip()

            if not remainder:
                return candidates[index - 1]

            if remainder == candidates[index - 1]:
                return candidates[index - 1]

    # Allow final output containing only candidate number.
    if re.fullmatch(r"\d+", value):
        index = int(value)

        if 1 <= index <= len(candidates):
            return candidates[index - 1]

    return None


def constrain_predictions(
    predictions: list[str],
    candidates: list[str],
):
    result = []

    for prediction in predictions:
        candidate = normalize_prediction(
            prediction,
            candidates,
        )

        if candidate and candidate not in result:
            result.append(candidate)

    return result[:5]


def format_history(history):
    if not history:
        return ""

    text = "\n\nTOOL HISTORY:\n"

    for item in history:
        text += (
            "\nAssistant:\n"
            + item["assistant"]
            + "\n\nTool result:\n"
            + item["tool"]
            + "\n"
        )

    return text


def run_agent4lr(
    backend,
    graph,
    condition,
    problem,
    failing_test,
    runtime_output,
    candidates,
    max_steps=12,
):
    candidate_text = "\n".join(
        f"{i}. {candidate}"
        for i, candidate in enumerate(
            candidates,
            1,
        )
    )

    evidence = f"""
SWE-BENCH CONDITION:
{condition}

SWE-BENCH PROBLEM STATEMENT:

{problem}

FAILING TEST:
{failing_test}

RUNTIME TEST OUTPUT:

{runtime_output}

CANDIDATE ENTITIES:

{candidate_text}
"""

    history = []
    inspected = []
    final_response = ""

    for step in range(1, max_steps + 1):

        print(
            f"\n[{condition}] Agent4LR STEP "
            f"{step}/{max_steps}",
            flush=True,
        )

        if step == max_steps:
            instruction = """
THIS IS THE FINAL STEP.

Do not call another tool.

Return exactly five ranked candidates using:

Top_1 : path/to/file.py::Entity
Top_2 : path/to/file.py::Entity
Top_3 : path/to/file.py::Entity
Top_4 : path/to/file.py::Entity
Top_5 : path/to/file.py::Entity
"""
        else:
            instruction = """
Inspect one candidate with
get_code_snippet_of_candidate(N)
or return the final Top_1..Top_5 ranking.
"""

        prompt = (
            evidence
            + format_history(history)
            + instruction
        )

        print(
            f"[{condition}] calling LLM...",
            flush=True,
        )

        response = backend.complete(
            SYSTEM,
            prompt,
        )

        final_response = response

        print(
            f"[{condition}] LLM RESPONSE:",
            flush=True,
        )
        print(
            response,
            flush=True,
        )

        raw_predictions = parse_top5(
            response
        )

        if raw_predictions:
            predictions = constrain_predictions(
                raw_predictions,
                candidates,
            )

            if len(predictions) == 5:
                return {
                    "condition": condition,
                    "predictions": predictions,
                    "steps": step,
                    "inspected_candidates": inspected,
                    "final_response": response,
                    "transcript": history,
                }

            history.append(
                {
                    "assistant": response,
                    "tool": (
                        "FINAL ANSWER REJECTED: "
                        "all five predictions must be "
                        "chosen exactly from the provided "
                        "candidate list."
                    ),
                }
            )

            continue

        index = parse_tool(
            response
        )

        if index is None:
            history.append(
                {
                    "assistant": response,
                    "tool": (
                        "FORMAT ERROR: use "
                        "get_code_snippet_of_candidate(N) "
                        "or return Top_1..Top_5."
                    ),
                }
            )

            continue

        if not (
            1 <= index <= len(candidates)
        ):
            history.append(
                {
                    "assistant": response,
                    "tool": (
                        "ERROR: candidate number must "
                        f"be between 1 and "
                        f"{len(candidates)}."
                    ),
                }
            )

            continue

        entity = candidates[
            index - 1
        ]

        print(
            f"[{condition}] TOOL: "
            f"candidate {index} -> {entity}",
            flush=True,
        )

        snippet = graph.snippet(
            entity
        )

        result = (
            f"Candidate {index}: {entity}\n\n"
            f"{snippet}"
        )

        print(
            f"[{condition}] TOOL RESULT:",
            flush=True,
        )

        print(
            result[:5000],
            flush=True,
        )

        inspected.append(
            {
                "candidate_number": index,
                "entity": entity,
            }
        )

        history.append(
            {
                "assistant": response,
                "tool": result,
            }
        )

    predictions = constrain_predictions(
        parse_top5(final_response),
        candidates,
    )

    return {
        "condition": condition,
        "predictions": predictions,
        "steps": max_steps,
        "inspected_candidates": inspected,
        "final_response": final_response,
        "transcript": history,
    }


def load_candidates(
    pool_data,
    condition,
):
    rows = pool_data[
        condition
    ]["candidates"]

    candidates = [
        row["entity"]
        for row in rows
    ]

    if len(candidates) != 20:
        raise ValueError(
            f"{condition}: expected 20 candidates, "
            f"found {len(candidates)}"
        )

    return candidates


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instance",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="qwen3.6:27b",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    root = (
        Path.home()
        / "AdaptiveContextOpt"
    )

    work = (
        root
        / "data"
        / "swebench_workspaces"
        / args.instance
    )

    outputs = (
        work
        / "outputs"
    )

    out = (
        outputs
        / "agent4lr"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = json.loads(
        (
            work
            / "metadata.json"
        ).read_text()
    )

    problem = metadata.get(
        "problem_statement",
        "",
    )

    fail_to_pass = metadata.get(
        "FAIL_TO_PASS",
        [],
    )

    if isinstance(
        fail_to_pass,
        list,
    ):
        failing_test = ", ".join(
            fail_to_pass
        )
    else:
        failing_test = str(
            fail_to_pass
        )

    raw_output = (
        outputs
        / "raw_test_output.txt"
    ).read_text(
        errors="replace"
    )

    lean_output = (
        outputs
        / "leanctx_test_output.txt"
    ).read_text(
        errors="replace"
    )

    pool_data = json.loads(
        (
            out
            / "candidate_pools.json"
        ).read_text()
    )

    raw_candidates = load_candidates(
        pool_data,
        "raw",
    )

    lean_candidates = load_candidates(
        pool_data,
        "leanctx",
    )

    graph = GraphifyIndex.from_json(
        work / "repo",
        work
        / "repo"
        / "graphify-out"
        / "graph.json",
    )

    backend = ChatBackend(
        provider="ollama",
        model=args.model,
        timeout=1800,
    )

    print()
    print(
        "============================================"
    )
    print(
        "RAW AGENT4LR"
    )
    print(
        "============================================"
    )

    raw_result = run_agent4lr(
        backend=backend,
        graph=graph,
        condition="RAW",
        problem=problem,
        failing_test=failing_test,
        runtime_output=raw_output,
        candidates=raw_candidates,
        max_steps=args.max_steps,
    )

    (
        out
        / "raw_agent4lr.json"
    ).write_text(
        json.dumps(
            raw_result,
            indent=2,
        )
    )

    print()
    print(
        "RAW FINAL TOP-5"
    )

    for i, entity in enumerate(
        raw_result["predictions"],
        1,
    ):
        print(
            f"Top-{i}: {entity}"
        )

    print()
    print(
        "============================================"
    )
    print(
        "LEANCTX AGENT4LR"
    )
    print(
        "============================================"
    )

    lean_result = run_agent4lr(
        backend=backend,
        graph=graph,
        condition="LEANCTX",
        problem=problem,
        failing_test=failing_test,
        runtime_output=lean_output,
        candidates=lean_candidates,
        max_steps=args.max_steps,
    )

    (
        out
        / "leanctx_agent4lr.json"
    ).write_text(
        json.dumps(
            lean_result,
            indent=2,
        )
    )

    print()
    print(
        "LEANCTX FINAL TOP-5"
    )

    for i, entity in enumerate(
        lean_result["predictions"],
        1,
    ):
        print(
            f"Top-{i}: {entity}"
        )

    result = {
        "instance_id": args.instance,
        "model": args.model,
        "gold_used_by_agent": False,
        "raw": raw_result,
        "leanctx": lean_result,
    }

    pair_file = (
        out
        / "agent4lr_pair.json"
    )

    pair_file.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(
        "============================================"
    )
    print(
        "DONE"
    )
    print(
        "============================================"
    )

    print(
        "Saved:",
        pair_file,
    )


if __name__ == "__main__":
    main()
