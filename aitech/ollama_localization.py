"""Run the same local Ollama fault-localization model on matched AITECH contexts.

Uses archived SymPy tool-output evidence, not a fresh SWE-bench execution.
This exploratory *localization-only* test does NOT apply patches. Requires an
Ollama server; any gold labels are used only in the evaluator after inference.

Example:
  python -m aitech.ollama_localization \\
    --model qwen3.6:27b --output results/aitech/localization.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .evidence import select_evidence
from .sympy_trace_pilot import DEFAULT_TRACE, DEPENDENCIES, extract

SYSTEM = (
    "You are a Python software fault localization assistant. "
    "Given an issue description and numbered source evidence, predict the "
    "single most suspicious file and function or class. "
    "Reply using JSON only with keys 'file', 'function', and 'reason'. "
    "Never invent source lines, and state uncertainty when necessary."
)
ISSUE = (
    "A SymPy Symbol unexpectedly has an instance __dict__ even though "
    "objects using __slots__ should not. Identify a likely source of the bug."
)
# Gold is deliberately absent from prompts and Ollama API call.
# Predictions are scored in a separate evaluation step.
GOLD_FILE = "sympy/core/_print_helpers.py"
GOLD_ENTITY = "Printable"

def make_context(items, selection) -> str:
    by_id = {item.id: item for item in items}
    return "\n".join(
        f"[{identifier}] {by_id[identifier].text}"
        for identifier in selection.ids
    )

def call_ollama(url: str, model: str, prompt: str, timeout: int, seed: int) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0, "seed": seed},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    result["measured_elapsed_seconds"] = time.monotonic() - started
    return result

def parse_prediction(result: dict) -> dict:
    content = result.get("message", {}).get("content", "")
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return {"file": "", "function": "", "reason": "", "parse_ok": False,
                "raw_response": content}
    if not isinstance(payload, dict):
        return {"file": "", "function": "", "reason": "", "parse_ok": False,
                "raw_response": content}
    return {
        "file": str(payload.get("file", "")),
        "function": str(payload.get("function", "")),
        "reason": str(payload.get("reason", "")),
        "parse_ok": True,
        "raw_response": content,
    }

def score_offline(prediction: dict) -> dict:
    """Offline-only scoring. Must never be included in the inference prompt."""
    file_name = prediction["file"].strip()
    symbol_name = prediction["function"].strip()
    return {
        "file_hit_1": file_name == GOLD_FILE,
        "entity_hit_1": symbol_name == GOLD_ENTITY or symbol_name.endswith("." + GOLD_ENTITY),
    }

def experiment(
    trace: str,
    *,
    model: str,
    endpoint: str = "http://127.0.0.1:11434",
    budget: int | None = None,
    timeout: int = 180,
    seed: int = 42,
    infer=call_ollama,
) -> dict:
    items = extract(trace)
    full = sum(x.tokens for x in items)
    budget = (full * 75) // 100 if budget is None else budget
    out = {
        "instance": "sympy__sympy-20590",
        "scope": "small archived-trace evidence microcase, not full SWE-bench replay",
        "model": model, "seed": seed, "temperature": 0,
        "budget_proxy": budget, "raw_proxy_tokens": full,
        "conditions": {},
    }
    for name, preserve in (("raw_excerpt", None),
                           ("relevance_only", False),
                           ("dependency_aware", True)):
        limit = full if preserve is None else budget
        selected = select_evidence(items, DEPENDENCIES, limit,
                                   preserve_dependencies=bool(preserve))
        context = make_context(items, selected)
        prompt = f"Issue:\n{ISSUE}\n\nEvidence:\n{context}\n"
        result = infer(endpoint, model, prompt, timeout, seed)
        prediction = parse_prediction(result)
        out["conditions"][name] = {
            "selected_ids": list(selected.ids),
            "tokens_proxy": selected.tokens,
            "dangling_ids": list(selected.dangling),
            "prompt_chars": len(prompt),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
            "elapsed_seconds": result.get("measured_elapsed_seconds"),
            "prediction": prediction,
            "offline_score": score_offline(prediction),
        }
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = experiment(
            args.trace.read_text(encoding="utf-8"),
            model=args.model, endpoint=args.endpoint, budget=args.budget,
            timeout=args.timeout, seed=args.seed,
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        parser.exit(2, f"Ollama connection failed: {exc}\n")
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

if __name__ == "__main__":
    main()
