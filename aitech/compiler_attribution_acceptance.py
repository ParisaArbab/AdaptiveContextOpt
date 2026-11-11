"""End-to-end compiler-to-evidence identity acceptance test.

Uses full bundled Java source, JDK semantic method bindings and ranked
candidate identifiers for a controlled fixture. This is NOT a Defects4J
localization experiment and no model or patch labels are involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .java_compiler_graph import audit_source_directory
from .verified_graph_evidence import map_calls

SOURCE_ROOT = Path(__file__).parent / "fixtures" / "java_compiler_verified"
OWNER = "aitech.fixture.CallerOverloadDemo"
CANDIDATES = (
    f"{OWNER}.use(String)",
    f"{OWNER}.use(Object)",
    f"{OWNER}.convert(String)",
    f"{OWNER}.convert(Object)",
)
EXPECTED_EDGES = ((0, 2), (1, 3))


def run(source_root: Path = SOURCE_ROOT) -> dict:
    audit = audit_source_directory(source_root)
    relevant = [
        call for call in audit.calls
        if call.caller_owner == OWNER and call.caller_method == "use"
        and call.target_owner == OWNER and call.target_method == "convert"
    ]
    mapped = map_calls(list(CANDIDATES), relevant)
    if mapped.edges != EXPECTED_EDGES:
        raise AssertionError(
            f"Compiler-to-candidate mapping did not match controlled fixture: {mapped.edges}"
        )
    if mapped.ambiguous_callers or mapped.ambiguous_targets:
        raise AssertionError("Controlled overload mapping was ambiguous")
    return {
        "scope": "Fully compilable controlled Java fixture, not Defects4J",
        "status": "compiler_attributed_and_candidate_mapped",
        "source_hashes": audit.source_hashes,
        "total_compiler_calls": len(audit.calls),
        "overloaded_calls_examined": len(relevant),
        "candidate_ids": list(CANDIDATES),
        "mapped_edges": [list(edge) for edge in mapped.edges],
        "mapping_ambiguity": {
            "callers": list(mapped.ambiguous_callers),
            "targets": list(mapped.ambiguous_targets),
            "unmatched": list(mapped.unmatched_targets),
        },
        "gold_used": False,
        "llm_executed": False,
        "limitations": [
            "Synthetic test source, not independent benchmark evidence",
            "Java static binding, not runtime dynamic dispatch",
            "No inference latency, contextual costs, localization or patch scoring",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/compiler_attribution_acceptance.json"))
    args = parser.parse_args()
    result = run(args.sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(result['mapped_edges'])} caller+callee overload mappings")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
