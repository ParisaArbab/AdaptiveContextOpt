"""Frozen EGAV 10-bug development validation from GRAFT batches 01 and 02.

No change to model selection rules. Live inference only with --allow-inference.
Usage: python research_method_agent/egav_validation.py preflight|run-01|run-02|report|status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "egav_validation_10bugs"
PILOT = {"Chart-17", "Closure-165", "Lang-28", "Math-95", "Time-26"}
# Git blob SHA checks guard the actual policy and evaluation algorithm, not a version label.
FROZEN_BLOBS = {
    "research_method_agent/egav_policies.py":
        "9cb908fc855bd15e93adf8e3172bc0e07becfb5e",
    "research_method_agent/research_method_agent.py":
        "10d4507d09406e77f23df1dc08ef7d721348d7ec",
}
COMPARE = ("l1", "code_cards", "risk_full", "code_cards_risk_full",
           "card_rank_full", "card_failure_full", "card_stop_full", "card_egav")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_policy_check():
    for name, expected in FROZEN_BLOBS.items():
        data = (ROOT / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != expected:
            raise RuntimeError("Frozen policy/code changed: " + name + ". Refusing validation.")


def config(batch):
    path = ROOT / "data/graft_100_cohort_20261115/batches" / (
        "batch_" + batch) / "experiment/config.json"
    data = read_json(path)
    bugs = data["benchmark"]["bugs"]
    if len(bugs) != 5 or len(set(bugs)) != 5 or set(bugs) & PILOT:
        raise ValueError("Expected five unique non-pilot bugs: " + str(bugs))
    dataset_file = Path(data["benchmark"]["dataset_file"])
    if not dataset_file.exists():
        raise FileNotFoundError("Original AIScalar prepared manifest missing: " +
                                str(dataset_file))
    return path, bugs


def invoke(batch, inference, args):
    path, bugs = config(batch)
    destination = RESULTS / ("batch_" + batch)
    command = [
        sys.executable,
        str(ROOT / "research_method_agent/research_method_agent.py"),
        "study", "--config", str(path), "--output-dir", str(destination),
        "--bugs", ",".join(bugs), "--candidate-limit", "24",
    ]
    if inference:
        if not args.allow_inference:
            raise ValueError("Real Ollama calls require --allow-inference")
        command += [
            "--allow-inference",
            "--ollama-url", args.ollama_url,
            "--l1-model", "qwen2.5:7b",
            "--full-model", "qwen2.5:7b",
            "--max-calls", str(args.max_calls),
            "--physical-token-cap", str(args.physical_token_cap),
            "--budget", "60000", "--max-verify", "2",
        ]
    print(("Collecting live responses: " if inference else "Preparing: ") +
          batch + " " + ",".join(bugs), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def summarize():
    results = []
    audit = []
    observed_bugs = set()
    for batch in ("01", "02"):
        directory = RESULTS / ("batch_" + batch)
        saved = read_json(directory / "evaluation/results.json")
        observation = read_json(directory / "observations.json")
        manifest = read_json(directory / "manifest.json")
        bug_ids = {bug["id"] for bug in manifest["bugs"]}
        if len(bug_ids) != 5 or observed_bugs & bug_ids:
            raise ValueError("Missing or overlapping bug IDs")
        observed_bugs |= bug_ids
        if saved.get("provenance") != "live_ollama":
            raise ValueError("Not a live Qwen observation set")
        if any(r.get("gold_level") != "file_proxy" for r in saved["results"]):
            raise ValueError("Mixed file-level and method-level gold")
        results.extend(saved["results"])
        attempts = observation.get("attempts", [])
        audit.append({
            "batch": batch,
            "bugs": sorted(bug_ids),
            "recorded_physical_tokens": sum(
                x["input_tokens"] + x["output_tokens"] for x in attempts
                if x.get("usage_known")),
            "physical_accounting_complete":
                observation.get("physical_accounting_complete", True),
            "unknown_usage_attempts":
                sum(not x.get("usage_known", False) for x in attempts),
        })
    if len(observed_bugs) != 10:
        raise ValueError("Expected 10 distinct validation bugs")
    lookup = {(r["bug"], r["arm"]): r for r in results}
    summary = []
    for arm in COMPARE:
        paired = []
        for bug in sorted(observed_bugs):
            base, tested = lookup.get((bug, "code_cards")), lookup.get((bug, arm))
            if base and tested and base["status"] == "complete" and tested["status"] == "complete":
                paired.append((base, tested))
        record = {"arm": arm, "paired_n": len(paired),
                  "complete": len(paired) == 10}
        if paired:
            record.update(
                mrr=statistics.mean(y["mrr"] for _, y in paired),
                delta_mrr=statistics.mean(y["mrr"] - x["mrr"] for x, y in paired),
                tokens=statistics.mean(y["tokens"] for _, y in paired),
                delta_tokens=statistics.mean(y["tokens"] - x["tokens"] for x, y in paired),
                win_tie_loss=[
                    sum(y["mrr"] > x["mrr"] for x, y in paired),
                    sum(y["mrr"] == x["mrr"] for x, y in paired),
                    sum(y["mrr"] < x["mrr"] for x, y in paired),
                ],
            )
        summary.append(record)
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "egav-frozen-validation-v1",
               "label_level": "file_proxy",
               "note": "Development validation, not untouched final test",
               "frozen_blobs": FROZEN_BLOBS,
               "physical_audit": audit, "summary": summary}
    (RESULTS / "combined_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# EGAV, 10 independent development bugs", "",
             "Fixed policy, real Qwen responses, offline replay, file-level proxy labels.",
             "The ten bugs were not used in the original five-bug pilot.", "",
             "| Policy | Paired bugs | MRR | Change vs cards | Tokens / bug | Token change |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        if not r["paired_n"]:
            lines.append("| " + r["arm"] + " | 0/10 | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| {r['arm']} | {r['paired_n']}/10 | {r['mrr']:.4f} | "
                f"{r['delta_mrr']:+.4f} | {r['tokens']:.0f} | "
                f"{r['delta_tokens']:+.0f} |")
    lines += ["", "Incomplete arms must not be described as validated.",
              "Logical policy costs do not include collecting unused counterfactual responses.",
              "Physical collection accounting: " + json.dumps(audit)]
    report = "\n".join(lines) + "\n"
    (RESULTS / "combined_report.md").write_text(report, encoding="utf-8")
    print(report)
    print("Saved:", RESULTS / "combined_report.md")


def status():
    for batch in ("01", "02"):
        path = RESULTS / ("batch_" + batch)
        m, o = path / "manifest.json", path / "observations.json"
        print("Batch", batch, "prepared:", m.exists())
        if m.exists():
            print(" Bugs:", [b["id"] for b in read_json(m)["bugs"]])
        if o.exists():
            obs = read_json(o)
            print(" Valid observations:", len(obs.get("observations", [])),
                  "Logged attempts:", len(obs.get("attempts", [])))
            print(" Complete physical accounting:",
                  obs.get("physical_accounting_complete", True))
        print(" Evaluation:", (path / "evaluation/results.json").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run-01", "run-02", "report", "status"))
    parser.add_argument("--allow-inference", action="store_true")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-calls", type=int, default=450)
    parser.add_argument("--physical-token-cap", type=int, default=800000)
    args = parser.parse_args()
    frozen_policy_check()
    if args.max_calls < 1 or args.physical_token_cap < 1:
        parser.error("Inference limits must be positive")
    if args.command == "preflight":
        for batch in ("01", "02"):
            invoke(batch, False, args)
    elif args.command in ("run-01", "run-02"):
        invoke(args.command[-2:], True, args)
    elif args.command == "report":
        summarize()
    else:
        status()


if __name__ == "__main__":
    main()
