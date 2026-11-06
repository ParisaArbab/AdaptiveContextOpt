"""Run a tiny controlled AITECH evidence-preservation demonstration.

Usage: python -m aitech.demo
The fixture is synthetic. No benchmark or ML accuracy is claimed.
"""
import json

from .evidence import Evidence, select_evidence


def run_demo() -> dict:
    evidence = [
        Evidence("failed_test", "test_parse_expression raises AssertionError", 4, 10.0),
        Evidence("call_bridge", "test_parse_expression invokes parse_expression", 4, 1.0),
        Evidence("unrelated", "unrelated deprecation warning", 4, 5.0),
    ]
    dependencies = {"failed_test": ("call_bridge",)}
    cases = {}
    for label, preserve in (("relevance_only", False), ("dependency_aware", True)):
        result = select_evidence(evidence, dependencies, budget=8, preserve_dependencies=preserve)
        cases[label] = {
            "selected_ids": list(result.ids),
            "selected_tokens": result.tokens,
            "dangling_ids": list(result.dangling),
            "complete": len(result.dangling) == 0,
        }
    return {
        "description": "Synthetic example, not a real fault localization benchmark",
        "budget": 8,
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
