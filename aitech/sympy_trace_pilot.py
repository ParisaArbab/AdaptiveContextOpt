"""AITECH offline pilot using actual saved Agent4SR SymPy trace.

This measures evidence selection ONLY; it does not call an LLM or imply
new file/function localization accuracy. Gold data is never read.
Run: python -m aitech.sympy_trace_pilot
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .evidence import Evidence, select_evidence

DEFAULT_TRACE = Path("data/swebench_workspaces/sympy__sympy-20590/outputs/compression_tax/agent4sr_qwen_COMPRESSION_TAX.log")
# Each anchor is a verbatim line visible in the historical agent trace.
# This is a deliberately small real-source microcase, not all trace lines.
ANCHORS = (
    ("test_assertion", "assert not hasattr(b1, '__dict__')", 10.0),
    ("test_function", "def test_immutable():", 3.0),
    ("object_constructor", "b1 = Basic()", 3.0),
    ("basic_definition", "class Basic(Printable, metaclass=ManagedProperties):", 7.0),
    ("printable_import", "from ._print_helpers import Printable", 1.0),
    ("sympify_import", "from .sympify import _sympify, sympify, SympifyError", 2.0),
    ("cache_import", "from .cache import cacheit", 1.0),
)
# Direct, manually inspected support links in historical source fragments.
# They are *structural* evidence links, not a proof of the bug's cause.
DEPENDENCIES = {
    "test_assertion": ("object_constructor", "test_function"),
    "object_constructor": ("basic_definition",),
    "basic_definition": ("printable_import",),
}

def approximate_tokens(text: str) -> int:
    """Documented regex proxy, not the downstream model tokenizer."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

def extract(trace: str) -> list[Evidence]:
    """Fail closed when any real-source anchor cannot be found."""
    results = []
    for identifier, anchor, relevance in ANCHORS:
        # Match complete source text lines, allowing the original numeric
        # source-line prefix in get_code_snippet tool results.
        matches = [line for line in trace.splitlines()
                   if line.strip().split(": ", 1)[-1].strip() == anchor]
        if not matches:
            raise ValueError(f"Missing trace anchor: {identifier} ({anchor})")
        results.append(Evidence(identifier, anchor, approximate_tokens(anchor), relevance))
    return results

def run_pilot(trace: str, budget: int | None = None) -> dict:
    evidence = extract(trace)
    total = sum(e.tokens for e in evidence)
    if budget is None:
        budget = (total * 75) // 100
    outputs = {}
    for name, preserve in (("relevance_only", False), ("dependency_aware", True)):
        selection = select_evidence(evidence, DEPENDENCIES, budget,
                                    preserve_dependencies=preserve)
        outputs[name] = {
            "selected": list(selection.ids),
            "tokens_proxy": selection.tokens,
            "dangling_selected": list(selection.dangling),
            "known_dependency_violations": len(selection.dangling),
            "selection_coverage": len(selection.ids) / len(evidence),
        }
    return {
        "instance": "sympy__sympy-20590",
        "evaluation_type": "offline_real_trace_microcase_no_LLM",
        "evidence_source": str(DEFAULT_TRACE),
        "ground_truth_used_for_selection": False,
        "score_source": "fixed manually defined research weights, NOT trained",
        "tokenizer": "regex proxy, NOT model-token counts",
        "evidence_items": len(evidence),
        "raw_tokens_proxy": total,
        "budget_tokens_proxy": budget,
        "conditions": outputs,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_pilot(args.trace.read_text(encoding="utf-8"), args.budget)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

if __name__ == "__main__":
    main()
