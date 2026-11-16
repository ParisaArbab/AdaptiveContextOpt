"""Prepare a single-run graph-specific ablation from an existing anchored config.

The same candidate pool, failure evidence, model and shared cache are used for:
  1. l1_anchor: L1 + failure-name ranking only; no graph or Full request.
  2. l1_graph: identical L1 + matching, then graph evidence.
  3. graft_adaptive: identical L1 + matching, graph and adaptive Full decisions.
No gold labels enter online decisions. Prior existing runs remain untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(source: Path, output: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text())
    benchmark = value["benchmark"]
    graft = value["graft"]
    if benchmark["batch_size"] != 1 or benchmark["model"] != "qwen2.5:7b":
        raise ValueError("Only the verified Qwen2.5 7B single-candidate experiment is supported")
    if not graft.get("failure_name_anchor"):
        raise ValueError("Source config must enable failure-name matching")
    if graft["max_graph"] < 1 or graft["max_full"] < 1:
        raise ValueError("Source config must allow graph and adaptive Full actions")
    if graft["max_full"] != benchmark["max_full"]:
        raise ValueError("Full action caps must agree")
    if output.exists():
        raise FileExistsError("Refusing to overwrite pre-existing ablation output: " + str(output))
    output.mkdir(parents=True)
    value["benchmark"]["policies"] = ["fixed_l1"]
    value["benchmark"]["run_id"] = "graft_graph_specific_" + output.name
    value["graft_policies"] = ["l1_anchor", "l1_graph", "graft_adaptive"]
    (output / "config.json").write_text(json.dumps(value, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "source_config": str(source.resolve()),
        "source_config_is_historical": True,
        "comparisons": [
            "l1_anchor: initial L1 plus failure-name matching; no graph calls",
            "l1_graph: same initial L1 plus graph refinements",
            "graft_adaptive: same initial L1 plus graph and adaptive Full decisions",
        ],
        "research_status": "exploratory_ablations",
        "all_policies_share_one_provider_journal": True,
    }, indent=2) + "\n")
    print("Prepared:", output)
    print("Bugs:", value["benchmark"]["bugs"])
    print("Policies:", value["benchmark"]["policies"] + value["graft_policies"])
    return output / "config.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
