"""Offline audit of invalid real LLM responses, no inference and no relabeling.

Uses saved per-policy CallResult data and strict parser. Shows whether errors
are caused by missing IDs, output truncation, malformed JSON, or schema issues.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys



def analyze(root: Path):
    errors = defaultdict(Counter)
    printed = set()
    for path in sorted((root / "instances").glob("*/*rrf*.json")):
        row = json.loads(path.read_text())
        policy = row.get("policy", "")
        if policy == "prior":
            continue
        bug = row.get("instance_id")
        for call in row.get("arm", {}).get("calls", []):
            if not call.get("error"):
                continue
            stage = call.get("stage", call.get("action", "unknown"))
            key = (bug, policy, stage)
            errors[key][str(call["error"])] += 1
            unique = (bug, stage, str(call["error"]))
            if unique in printed:
                continue
            printed.add(unique)
            raw = call.get("raw_response", "")
            output = call.get("output_tokens")
            try:
                value = json.loads(raw)
                keys = sorted(value) if isinstance(value, dict) else [type(value).__name__]
                result = value.get("results") if isinstance(value, dict) else None
                got_ids = [r.get("id") for r in result if isinstance(r, dict)] if isinstance(result, list) else None
            except (ValueError, TypeError):
                keys, got_ids = None, None
            print("EXAMPLE",bug,policy,stage)
            print(" error:",call["error"],"output_tokens:",output)
            print(" raw_bytes:",len(raw.encode()),"json_keys:",keys,"returned_ids:",got_ids)
            print(" raw_tail:",repr(raw[-160:]))
    for key, count in sorted(errors.items()):
        print("SUMMARY",key,dict(count))
    if not errors:
        print("No invalid paid call responses found")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/audit_graft_invalid_responses.py PATH_TO_CONTROL_RUN")
    analyze(Path(sys.argv[1]))
