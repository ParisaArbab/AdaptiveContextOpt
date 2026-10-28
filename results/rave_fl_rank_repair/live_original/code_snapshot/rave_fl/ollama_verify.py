"""Actual Ollama verifier with per-call usage and conservative budget admission.

Provide method source as JSON; missing source is never silently invented.
The estimated preflight bound is not a tokenizer-based hard guarantee.
"""
from __future__ import annotations
import json
import time
import urllib.request
from dataclasses import dataclass
from rave_fl.core import approx_tokens


@dataclass
class Result:
    method: str
    suspicion: float
    explanation: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float


SYSTEM = ("You are a Java fault localization verifier. Given the failing-test "
          "evidence and a candidate Java method, estimate whether it causes "
          "the failure. Respond ONLY with JSON: "
          '{"suspicion":0.0,"explanation":"short reason"}. '
          "Do not assume the method is faulty solely because it is listed.")


def verify(base_url: str, model: str, bug_context: str, method: str,
           source: str, max_output_tokens: int = 96, timeout: int = 180) -> Result:
    user = f"FAILURE CONTEXT:\n{bug_context}\n\nMETHOD:\n{method}\n\nSOURCE:\n{source}"
    payload = json.dumps({
        "model": model, "stream": False, "think": False,
        "format": "json",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0, "num_predict": max_output_tokens}
    }).encode()
    start = time.monotonic()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat", payload,
        {"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    duration = time.monotonic() - start
    if "prompt_eval_count" not in data or "eval_count" not in data:
        raise ValueError("Ollama response lacks actual token usage counters")
    message = data.get("message", {}).get("content", "")
    parsed = json.loads(message)
    score = float(parsed["suspicion"])
    if not 0 <= score <= 1:
        raise ValueError("Invalid suspicion score")
    return Result(method, score, str(parsed.get("explanation", "")),
                  int(data["prompt_eval_count"]), int(data["eval_count"]), duration)


def admission_estimate(context: str, method: str, source: str, output_cap: int) -> int:
    # Conservative planning only. Final actual model tokens may exceed it.
    return 2 * approx_tokens(SYSTEM + context + method + source) + output_cap + 96
