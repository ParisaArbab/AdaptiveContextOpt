"""Seal the archived native token audit for the fixed two-defect offline preflight."""
import argparse
import hashlib
import json
from pathlib import Path

from rave_fl.eval.schema import file_hash, fingerprint
from rave_fl.graft.prompt_settlement import BUGS, COHORTS, MODEL_DIGEST, RUNNER, SCHEMA, TEMPLATE, VERSION, require
from rave_fl.pilot import write_json


def rendered(messages):
    require([row["role"] for row in messages] == ["system", "user"], "Unexpected audit request shape")
    body = "".join(f"<|start_header_id|>{row['role']}<|end_header_id|>\n\n{row['content']}<|eot_id|>"
                   for row in messages)
    return (body + "<|start_header_id|>assistant<|end_header_id|>\n\n").encode()


def build(audit, base_config, certificate, output_config, bugs=tuple(BUGS)):
    require(tuple(bugs) in COHORTS, "Select Lang-54 or the full Lang-54 Math-80 cohort")
    read = lambda path: json.loads(path.read_text())
    plan = read(audit / "paired-plan.json")
    native = read(audit / "native-comparison.json")
    summary = read(audit / "readiness-summary.json")
    hashes = read(audit / "frozen-input-receipt.json")["hashes"]
    for name in ("native-comparison.json", "paired-plan.json"):
        expected, = [h for p, h in hashes.items() if Path(p).name == name]
        require(file_hash(audit / name) == expected, f"Archived {name} checksum mismatch")
    log = audit / "native-tokenizer.log"
    require("vocab only - skipping tensors" in log.read_text() and not native["mismatches"]
            and native["model_sha256"] == MODEL_DIGEST and native["template_sha256"] == TEMPLATE
            and [pair["instance_id"] for pair in plan["pairs"]] == list(BUGS)
            and plan["new_model_calls"] == 0, "Missing or incompatible native tokenizer audit")
    by_hash = {row["rendered_sha256"]: row for row in native["rows"]}
    require(len(by_hash) == native["requests"] == 4589 and all(r["matches"] for r in by_hash.values()),
            "Incomplete or mismatched native token capture")
    requests, pairs = [], {}
    for pair in plan["pairs"]:
        bug = pair["instance_id"]
        if bug not in bugs:
            continue
        pairs[bug] = {**{k: pair[k] for k in ("pool_sha256", "graph_sha256", "embeddings_sha256")},
                      "schedule_sha256": fingerprint(pair["schedule"])}
        for arm, rows in pair["arms"].items():
            require(len(rows) == BUGS[bug], "Audited arm is incomplete")
            for row in rows:
                payload = row["request"]
                record = by_hash[hashlib.sha256(rendered(payload["messages"])).hexdigest()]
                ids = record["token_ids"]
                require(row["status"] == "ready" and record["tokens"] == len(ids) == row["input_units"]
                        and ids[0] == 128000 and all(type(token) is int and 0 <= token < 128256 for token in ids),
                        "Invalid audited request or token IDs")
                requests.append({"instance_id": bug, "arm": arm, "target": row["target"],
                                 "request_sha256": fingerprint(payload), "prompt_tokens": len(ids),
                                 "token_ids_sha256": fingerprint(ids)})
    for path in (certificate, output_config):
        require(not path.exists(), "Choose a fresh certificate and configuration output")
    write_json(certificate, {"schema": SCHEMA, "serving": {
        "model_manifest_sha256": MODEL_DIGEST, "template_sha256": TEMPLATE,
        "serving_version": VERSION, "runner": RUNNER, "system": "Darwin", "architecture": "arm64",
        "binary_sha256": native["serving_binary_sha256"],
        "model_blob_sha256": native["model_blob_sha256"].removeprefix("sha256:"),
        "native_audit_sha256": file_hash(audit / "native-comparison.json"),
        "native_tokenizer_log_sha256": file_hash(log),
    }, "dataset_sha256": plan["dataset"]["dataset_sha256"],
        "tokenizer_files": summary["tokenizer_files"], "pairs": pairs, "requests": requests})
    config = read(base_config)
    config["benchmark"].update(bugs=list(bugs), dataset_file="data/graft_real_preparation/online/validated-two-defects.json")
    config["tokenizer_dir"] = str(audit / "tokenizer")
    config["prompt_settlement"] = {"file": str(certificate), "sha256": file_hash(certificate)}
    write_json(output_config, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "base-config", "certificate", "output-config"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--bugs", nargs="+", choices=BUGS, default=tuple(BUGS))
    args = parser.parse_args()
    build(args.audit, args.base_config, args.certificate, args.output_config, args.bugs)


if __name__ == "__main__":
    main()
