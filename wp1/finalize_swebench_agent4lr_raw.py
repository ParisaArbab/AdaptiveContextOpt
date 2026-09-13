from __future__ import annotations

import json
from pathlib import Path

from wp1.llm_backends import ChatBackend
from wp1.run_swebench_agent4lr_pair import (
    SYSTEM,
    parse_top5,
    constrain_predictions,
    format_history,
)


INSTANCE = "sympy__sympy-20590"

root = Path.home() / "AdaptiveContextOpt"

work = (
    root
    / "data"
    / "swebench_workspaces"
    / INSTANCE
)

outputs = work / "outputs"
out = outputs / "agent4lr"

pair_path = out / "agent4lr_pair.json"
raw_path = out / "raw_agent4lr.json"

pair = json.loads(pair_path.read_text())
raw_result = json.loads(raw_path.read_text())

pool = json.loads(
    (out / "candidate_pools.json").read_text()
)

candidates = [
    row["entity"]
    for row in pool["raw"]["candidates"]
]

if len(candidates) != 20:
    raise RuntimeError(
        f"Expected 20 RAW candidates, found {len(candidates)}"
    )

metadata = json.loads(
    (work / "metadata.json").read_text()
)

problem = metadata.get(
    "problem_statement",
    "",
)

fail_to_pass = metadata.get(
    "FAIL_TO_PASS",
    [],
)

if isinstance(fail_to_pass, list):
    failing_test = ", ".join(fail_to_pass)
else:
    failing_test = str(fail_to_pass)

runtime_output = (
    outputs / "raw_test_output.txt"
).read_text(
    errors="replace"
)

candidate_text = "\n".join(
    f"{i}. {candidate}"
    for i, candidate in enumerate(
        candidates,
        1,
    )
)

history = raw_result.get(
    "transcript",
    [],
)

prompt = f"""
SWE-BENCH CONDITION:
RAW

SWE-BENCH PROBLEM STATEMENT:

{problem}

FAILING TEST:
{failing_test}

RUNTIME TEST OUTPUT:

{runtime_output}

CANDIDATE ENTITIES:

{candidate_text}

{format_history(history)}

THE INVESTIGATION IS NOW COMPLETE.

Do NOT call another tool.
Do NOT search for any new candidate.
Do NOT propose a patch.

Using only the evidence already collected, choose exactly five entities
from the candidate list.

Your response MUST contain exactly these five lines and nothing else:

Top_1 : path/to/file.py::Entity
Top_2 : path/to/file.py::Entity
Top_3 : path/to/file.py::Entity
Top_4 : path/to/file.py::Entity
Top_5 : path/to/file.py::Entity
"""

model = pair.get(
    "model",
    "qwen3.6:27b",
)

backend = ChatBackend(
    provider="ollama",
    model=model,
    timeout=1800,
)

predictions = []
final_response = ""
success_attempt = None

for attempt in range(1, 4):

    print()
    print(
        "============================================"
    )
    print(
        f"RAW FINALIZATION ATTEMPT {attempt}/3"
    )
    print(
        "============================================"
    )

    response = backend.complete(
        SYSTEM,
        prompt,
    )

    final_response = response

    print(response)

    parsed = parse_top5(
        response
    )

    predictions = constrain_predictions(
        parsed,
        candidates,
    )

    print()
    print(
        "Valid candidates parsed:",
        len(predictions),
    )

    if len(predictions) == 5:
        success_attempt = attempt
        break

    prompt += f"""

Your previous final response was invalid because it did not produce
exactly five valid candidates from the supplied candidate list.

Previous invalid response:

{response}

Try again.

Return ONLY:

Top_1 : candidate
Top_2 : candidate
Top_3 : candidate
Top_4 : candidate
Top_5 : candidate
"""


if len(predictions) != 5:
    print()
    print(
        "ERROR: RAW Agent4LR still does not have "
        "five valid predictions."
    )
    raise SystemExit(2)


# Preserve invalid original outputs.
backup_raw = (
    out
    / "raw_agent4lr.INVALID_no_final_top5.json"
)

backup_pair = (
    out
    / "agent4lr_pair.BEFORE_RAW_FINALIZE_RETRY.json"
)

if not backup_raw.exists():
    backup_raw.write_text(
        raw_path.read_text()
    )

if not backup_pair.exists():
    backup_pair.write_text(
        pair_path.read_text()
    )


# Update RAW only.
fixed_raw = dict(raw_result)

fixed_raw["predictions"] = predictions
fixed_raw["final_response"] = final_response
fixed_raw["finalization_only_retry"] = True
fixed_raw["finalization_retry_attempts"] = success_attempt
fixed_raw["original_investigation_steps"] = raw_result.get(
    "steps"
)

raw_path.write_text(
    json.dumps(
        fixed_raw,
        indent=2,
    )
)

pair["raw"] = fixed_raw

pair_path.write_text(
    json.dumps(
        pair,
        indent=2,
    )
)


print()
print(
    "============================================"
)
print(
    "RAW FINAL TOP-5"
)
print(
    "============================================"
)

for i, entity in enumerate(
    predictions,
    1,
):
    print(
        f"Top-{i}: {entity}"
    )

print()
print(
    "LeanCTX result was NOT rerun or modified."
)

print(
    "RAW saved:",
    raw_path,
)

print(
    "Pair saved:",
    pair_path,
)
