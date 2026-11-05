"""Summarize incomplete GRAFT arms without inference or altered denominators."""
from collections import Counter
import json
from pathlib import Path
import sys


def diagnose(root):
    root = Path(root)
    for condition in ("control", "anchored"):
        directory = root / condition
        if not (directory / "results.json").exists():
            print(condition, "NOT EXECUTED")
            continue
        rows = json.loads((directory / "results.json").read_text())
        accounting = json.loads((directory / "accounting.json").read_text())
        print("\nCONDITION:", condition)
        print("Physical provider calls:", accounting.get("new_calls"))
        print("Physical provider tokens:", accounting.get("known_provider_tokens"))
        for row in sorted(rows, key=lambda r: (r.get("instance_id", ""), r.get("policy", ""))):
            arm = row.get("arm", {})
            states = arm.get("states", {})
            invalid = [key for key, state in states.items()
                       if not state.get("observation") or not state["observation"].get("valid")]
            stages = Counter(c.get("stage", "unknown") for c in arm.get("calls", []))
            metrics = row.get("metrics", {}).get("java_strict_v2", {}).get("method")
            print(
                row.get("instance_id"), row.get("policy"),
                "status:", row.get("status"),
                "valid:", len(states) - len(invalid), "/", len(states),
                "invalid:", len(invalid),
                "stages:", dict(stages),
                "stop:", arm.get("stop_reason"),
                "error:", row.get("error"),
                "MRR:", metrics.get("MRR") if metrics else None,
                "Top5:", metrics.get("Top-5") if metrics else None,
                "label_note:", row.get("gold_unavailable_reason"),
            )
            for ident in invalid[:2]:
                value = states[ident].get("observation") or {}
                print("  invalid example:", ident,
                      "reason:", value.get("reason"), "missing:", value.get("missing_evidence"))
        print("Note: incomplete arms cannot be counted as completed success.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/diagnose_graft_independent.py RUN_DIR")
    diagnose(sys.argv[1])
