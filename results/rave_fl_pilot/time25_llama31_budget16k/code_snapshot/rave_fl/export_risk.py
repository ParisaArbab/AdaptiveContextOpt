"""Reuse measured controlled cheap observations for OFFLINE risk labels.

No model calls: Full + initial L0 confidence discovery + fixed L1 batches.
Inference snapshots were saved before reading gold. Reuse avoids paying for
identical probes again. A partial baseline or partial cheap coverage is rejected.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rave_fl.pilot_evaluation import risk_rows
from wp1.flexfl_data import read_ground_truth


def export(run_path, manifest_path, output):
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete" or run.get("dry_run"):
        raise ValueError("Require a completed real run")
    if output.exists():
        raise ValueError("Refuse to overwrite an existing risk dataset")
    cases = {c["bug"]: c for c in json.loads(manifest_path.read_text())["cases"]}
    rows = []
    for case in run["cases"]:
        truths = read_ground_truth(Path(cases[case["bug"]]["flexfl_repo"]), case["bug"])
        rows.extend(risk_rows(case["bug"], case, truths))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    return {"rows": len(rows), "harmful": sum(r["label"] for r in rows),
            "bugs": sorted({r["bug"] for r in rows}), "new_llm_calls": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(export(a.run, a.manifest, a.output), indent=2))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
