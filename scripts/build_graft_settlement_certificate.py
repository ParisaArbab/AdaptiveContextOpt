"""Seal a supported fixed cohort from the archived native token audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.eval.schema import file_hash, fingerprint
from rave_fl.graft.prompt_settlement import (
    BGE_MODEL, BUGS, COHORTS, MAC_BINARY_SHA256, MODEL_DIGEST, RUNNER,
    SCHEMA, SCHEDULE_SCHEMA, TEMPLATE, VERSION, SettlementCertificate, require,
)
from rave_fl.pilot import write_json


def rendered(messages):
    require([row["role"] for row in messages] == ["system", "user"], "Unexpected audit request shape")
    body = "".join(f"<|start_header_id|>{row['role']}<|end_header_id|>\n\n{row['content']}<|eot_id|>"
                   for row in messages)
    return (body + "<|start_header_id|>assistant<|end_header_id|>\n\n").encode()


def build(audit, base_config, certificate, output_config, bugs=tuple(BUGS)):
    require(tuple(bugs) in COHORTS, "Select Lang-54, Math-80, or Lang-54 Math-80 in audited order")
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


def build_schedule(audit, base_config, certificate, output_config, *, native_sha256, log_sha256):
    plan_path = audit / "paired-plan.json"
    native_path, log_path = audit / "native-comparison.json", audit / "native-tokenizer.log"
    require(file_hash(native_path) == native_sha256 and file_hash(log_path) == log_sha256,
            "Native capture bytes differ from the supplied receipts")
    plan = json.loads(plan_path.read_text())
    native = json.loads(native_path.read_text())
    require(native["schema"] == "graft-native-schedule-audit-v1"
            and native["capture_plan_sha256"] == file_hash(plan_path)
            and (native["system"], native["architecture"], native["serving_binary_sha256"])
            == ("Darwin", "arm64", MAC_BINARY_SHA256)
            and native["model_sha256"] == MODEL_DIGEST and native["template_sha256"] == TEMPLATE
            and not native["mismatches"] and b"vocab only - skipping tensors" in log_path.read_bytes(),
            "Invalid native schedule capture, plan binding or vocabulary-only receipt")
    by_hash = {row["rendered_sha256"]: row for row in native["rows"]}
    require(len(by_hash) == len(native["rows"]) == native["requests"] > 0
            and all(row["matches"] is True for row in by_hash.values()), "Incomplete or mismatched native captures")
    settings = json.loads(base_config.read_text())
    config = fixed.configuration(settings)
    require(not config.get("prompt_settlement") and config["tokenizer_dir"] and plan["new_model_calls"] == 0
            and plan["config"] == config and plan["arm_diff"] == fixed.ARM_DIFF,
            "Base configuration differs from the provider-free fixed capture")
    tokenizer = Path(config["tokenizer_dir"])
    tokenizer_files = {name: file_hash(tokenizer / name) for name in ("tokenizer.json", "tokenizer_config.json")}
    require(all(plan["resources"].get(str(tokenizer / name)) == sha for name, sha in tokenizer_files.items()),
            "Tokenizer files differ from the captured plan")
    dataset = config["benchmark"]["dataset_file"]
    require(file_hash(dataset) == plan["dataset"]["dataset_sha256"], "Dataset bytes differ from the captured plan")
    pairs, schedules, requests = {}, {}, []
    encoder = None
    for pair in plan["pairs"]:
        bug = pair["instance_id"]
        require(bug not in pairs and pair["embedding_synthetic"] is False and pair["embedding_model_id"] == BGE_MODEL
                and pair["embedding_dimension"] == 1024, "Missing real pinned embeddings or duplicate defect")
        provenance = pair["embedding_provenance"]
        current = {k: provenance[k] for k in ("checkpoint_revision", "checkpoint_files", "max_length",
                                               "batch_size", "device_requested", "packages")}
        require(encoder is None or encoder == current, "Mixed encoder provenance in the cohort")
        encoder = current
        pairs[bug] = {**{k: pair[k] for k in ("pool_sha256", "graph_sha256", "embeddings_sha256")},
                      "schedule_sha256": fingerprint(pair["schedule"])}
        schedules[bug] = [row["target"] for row in pair["schedule"]]
        for arm in fixed.ARMS:
            rows = pair["arms"][arm]
            require([row["target"] for row in rows] == schedules[bug], "Incomplete or reordered capture arm")
            for row in rows:
                payload = row["request"]
                capture = by_hash.get(hashlib.sha256(rendered(payload["messages"])).hexdigest())
                require(capture and row["status"] == "ready" and row["exact_template_tokens"],
                        "Exact final request missing from native capture")
                ids = capture["token_ids"]
                require(capture["tokens"] == len(ids) == row["input_units"] <= 1024 and ids[0] == 128000
                        and all(type(token) is int and 0 <= token < 128256 for token in ids),
                        "Invalid exact prompt token sequence")
                requests.append({"instance_id": bug, "arm": arm, "target": row["target"],
                                 "request_sha256": fingerprint(payload), "prompt_tokens": len(ids),
                                 "token_ids_sha256": fingerprint(ids)})
    proof = {"schema": SCHEDULE_SCHEMA, "dataset_sha256": file_hash(dataset),
             "serving": {"model_manifest_sha256": MODEL_DIGEST, "template_sha256": TEMPLATE,
                         "serving_version": VERSION, "runner": RUNNER, "system": "Darwin", "architecture": "arm64",
                         "binary_sha256": MAC_BINARY_SHA256, "model_blob_sha256": native["model_blob_sha256"],
                         "capture_plan_sha256": native["capture_plan_sha256"],
                         "native_audit_sha256": native_sha256, "native_tokenizer_log_sha256": log_sha256},
             "tokenizer_files": tokenizer_files, "encoder": encoder, "pairs": pairs, "schedules": schedules,
             "requests": requests}
    require(certificate.resolve() != output_config.resolve()
            and not certificate.exists() and not output_config.exists(), "Choose distinct new certificate and config paths")
    write_json(certificate, proof)
    spec = {"file": str(certificate.resolve()), "sha256": file_hash(certificate)}
    certified = {**config, "prompt_settlement": spec}
    try:
        SettlementCertificate(certified)
        verified, _ = fixed.prepare(certified)
        require(verified["run_admitted"] and verified["planned_fresh_calls_upper"] == len(requests)
                and all(pair["status"] == "ready" for pair in verified["pairs"]),
                "Certificate cannot admit the complete paired cohort under the original caps")
    except Exception:
        certificate.unlink()
        raise
    write_json(output_config, {**settings, "prompt_settlement": spec})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "base-config", "certificate", "output-config"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--bugs", nargs="+", choices=BUGS, default=tuple(BUGS))
    parser.add_argument("--schedule", action="store_true", help="Seal a new immutable native-captured schedule")
    parser.add_argument("--native-sha256")
    parser.add_argument("--log-sha256")
    args = parser.parse_args()
    if args.schedule:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", GIT_ALLOW_PROTOCOL="",
                          GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0", TOKENIZERS_PARALLELISM="false")
        build_schedule(args.audit, args.base_config, args.certificate, args.output_config,
                       native_sha256=args.native_sha256, log_sha256=args.log_sha256)
    else:
        build(args.audit, args.base_config, args.certificate, args.output_config, args.bugs)


if __name__ == "__main__":
    main()
