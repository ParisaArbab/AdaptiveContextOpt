"""Run six synthetic fixed-evidence cases and one explicit admission rejection."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from rave_fl.eval.adapters import read_rows
from rave_fl.eval.schema import safe_name
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.fixed_evidence_reporting import report
from rave_fl.pilot import write_json


def rehearse(config, fixture_labels, output, physical_budget):
    if config["benchmark"]["backend"] != "fake":
        raise ValueError("The rehearsal requires the fake backend")
    output = Path(output).resolve()
    rows, _ = read_rows(config["benchmark"]["dataset_file"])
    labels, _ = read_rows(fixture_labels)
    by_id = {row["instance_id"]: row for row in rows}
    gold = {row["instance_id"]: row["gold"] for row in labels}
    required = {"synthetic-java", "synthetic-python"}
    if not required <= by_id.keys() or not required <= gold.keys():
        raise ValueError("Supply the prepared Java/Python fixture and its separate label file")
    java, python = by_id["synthetic-java"], by_id["synthetic-python"]
    new_rows = []
    new_labels = []
    output.mkdir(parents=True, exist_ok=False)
    embeddings = output / "embeddings"
    embeddings.mkdir()
    manifest_parent = Path(config["benchmark"]["dataset_file"]).resolve().parent
    cases = [
        ("covered-java", java, gold["synthetic-java"]),
        ("covered-python", python, gold["synthetic-python"]),
        ("missing-label", python, None),
        ("file-only-label", python, {**gold["synthetic-python"], "functions": []}),
        ("candidate-miss", python, {**gold["synthetic-python"], "functions": ["example.py::absent"]}),
        ("invalid-verdict", java, gold["synthetic-java"]),
    ]
    for name, source, label in cases:
        checkout = (manifest_parent / source["checkout"]).resolve()
        new_rows.append({**source, "instance_id": name, "checkout": str(checkout)})
        if label is not None:
            new_labels.append({"instance_id": name, "gold": label})
        shutil.copyfile(Path(config["embeddings_dir"]) / (safe_name(source["instance_id"]) + ".json"),
                        embeddings / (safe_name(name) + ".json"))
    dataset_file, labels_file = output / "instances.json", output / "labels.json"
    write_json(dataset_file, {"version": "phase4-synthetic-v1", "instances": new_rows})
    write_json(labels_file, {"instances": new_labels})
    value = {
        **config, "embeddings_dir": str(embeddings), "labels_file": str(labels_file),
        "benchmark": {**config["benchmark"], "dataset_file": str(dataset_file),
                      "bugs": [name for name, _, _ in cases], "limit": None,
                      "physical_budget": physical_budget},
    }
    write_json(output / "config.json", value)
    opts = fixed.configuration(value)
    plan, _ = fixed.prepare(opts)
    if (plan["planned_fresh_calls_upper"] != 24
            or any(pair["status"] != "ready" or len(pair["schedule"]) != 2 for pair in plan["pairs"])):
        raise ValueError(f"Explicit rehearsal caps or resources rejected a pair: {[p['reasons'] for p in plan['pairs']]}")
    invalid_index = next(i for i, pair in enumerate(plan["pairs"]) if pair["instance_id"] == "invalid-verdict")
    before_invalid = sum(pair["max_fresh_calls"] for pair in plan["pairs"][:invalid_index])
    calls = 0
    execute = fixed.GraftProvider.execute

    def send(provider, payload, ids, action):
        nonlocal calls
        if provider.kind != "fake":
            raise AssertionError("A real provider reached the fixture")
        response = execute(provider, payload, ids, action)
        if calls == before_invalid:
            response["message"]["content"] = "{}"
        calls += 1
        return response

    with patch.object(fixed.GraftProvider, "execute", send):
        result = fixed.run(opts, output / "run")
    resumed = fixed.run(opts, output / "run", resume=True)
    if calls != 24 or resumed != {**result, "new_calls_this_invocation": 0}:
        raise ValueError("Synthetic collection or resume differed from the rehearsal schedule")
    summary = report(output / "run", output / "report")

    denied = {**value, "per_arm_max_new_calls": 1,
              "benchmark": {**value["benchmark"], "bugs": ["covered-java"]}}
    write_json(output / "denied-config.json", denied)
    rejection = fixed.run(fixed.configuration(denied), output / "denied-run")
    if rejection["new_calls_this_invocation"] != 0:
        raise ValueError("The intentionally infeasible arm budget admitted a call")
    report(output / "denied-run", output / "denied-report")
    write_json(output / "rehearsal.json", {
        "synthetic": True, "scheduled_fixture_pairs": summary["selected_pairs"],
        "simulated_fresh_requests": calls, "new_requests_on_resume": resumed["new_calls_this_invocation"],
        "intentionally_denied_requests": rejection["new_calls_this_invocation"],
        "real_llm_requests": 0,
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-config", required=True, type=Path)
    parser.add_argument("--fixture-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical-budget", required=True, type=int,
                        help="Explicit whole-run simulated token cap for the six-case fixture")
    args = parser.parse_args()
    result = rehearse(json.loads(args.fixture_config.read_text()), args.fixture_labels, args.output,
                      args.physical_budget)
    print(f"Reported {result['selected_pairs']} synthetic pairs in {args.output}")


if __name__ == "__main__":
    main()
