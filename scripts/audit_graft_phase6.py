"""Provider-free readiness audits for pinned local Defects4J and serving artifacts."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

from rave_fl.eval.schema import file_hash, fingerprint
from rave_fl.graft.fixed_evidence_reporting import load_json, require
from rave_fl.graft.selection import ByteCounter, TokenizerCounter
from rave_fl.pilot import write_json
from wp1.flexfl_data import TRADITIONAL_METHODS, flexfl_root, read_ground_truth


BUGS = ("Lang-54", "Math-80", "Closure-61")
GIT_OFFLINE = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""}
SHA256 = re.compile(r"[0-9a-f]{64}")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], env=GIT_OFFLINE,
                          capture_output=True, check=False)


def local_blob(repo, revision, relative, target):
    entry = git(repo, "ls-tree", revision, "--", relative).stdout.decode().strip()
    if not entry:
        return {"status": "absent_from_tree"}
    _, kind, object_id = entry.split("\t", 1)[0].split()
    require(kind == "blob", f"Expected a metadata blob for {relative}")
    obj = git(repo, "cat-file", "blob", object_id)
    if obj.returncode:
        return {"status": "blob_unavailable_offline", "git_blob": object_id}
    require(hashlib.sha1(f"blob {len(obj.stdout)}\0".encode() + obj.stdout).hexdigest() == object_id,
            f"Metadata bytes disagree with pinned Git object {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(obj.stdout)
    return {"status": "verified_pinned_archive", "git_blob": object_id, "sha256": file_hash(target)}


def defects4j(args):
    require(re.fullmatch(r"[0-9a-f]{40}", args.revision) is not None
            and git(args.metadata, "cat-file", "-t", args.revision).stdout.strip() == b"commit",
            "Supply a locally available full Defects4J metadata commit")
    args.output.mkdir(parents=True, exist_ok=False)
    root = flexfl_root(args.flexfl)
    rows = []
    for bug in BUGS:
        project, ident = bug.split("-")
        prefix = f"framework/projects/{project}"
        target = args.output / bug
        files = {
            "revision_table": ("active-bugs.csv", "active-bugs.csv"),
            "trigger_trace": (f"trigger_tests/{ident}", "archived-trigger.txt"),
            "source_patch": (f"patches/{ident}.src.patch", "offline-source.patch"),
        }
        archived = {
            key: local_blob(args.metadata, args.revision, f"{prefix}/{relative}", target / filename)
            for key, (relative, filename) in files.items()
        }
        mapping = None
        if archived["revision_table"]["status"] == "verified_pinned_archive":
            entries = list(csv.DictReader(io.StringIO((target / "active-bugs.csv").read_text())))
            matching = [row for row in entries if row["bug.id"] == ident]
            require(len(matching) <= 1, f"Duplicate official rows for {bug}")
            mapping = matching[0] if matching else None
        checkout = args.checkouts / bug
        head = git(checkout, "rev-parse", "HEAD").stdout.decode().strip() if checkout.is_dir() else ""
        dirty = git(checkout, "status", "--porcelain", "--untracked-files=no") if head else None
        matched = bool(mapping and head == mapping["revision.id.buggy"]
                       and dirty and dirty.returncode == 0 and not dirty.stdout)
        candidates = {}
        for method in TRADITIONAL_METHODS:
            source = root / "data" / "FL_results" / method / "Defects4J" / f"{bug}_method-susps.csv"
            candidates[method] = file_hash(source) if source.is_file() else None
        try:
            gold = read_ground_truth(args.flexfl, bug)
        except FileNotFoundError:
            gold = []
        rows.append({
            "instance_id": bug, "status": "blocked", "archive": archived, "official_mapping": mapping,
            "local_checkout_head": head or None, "clean_exact_buggy_checkout": matched,
            "failure_replayed_in_phase": False, "candidate_csv_sha256": candidates,
            "candidate_pool_verified": False, "historical_flexfl_method_labels": gold,
            "method_gold_verified_on_buggy_source": False,
        })
    write_json(args.output / "defects4j-readiness.json", {
        "schema": "graft-defects4j-offline-audit-v1", "metadata_commit": args.revision,
        "metadata_origin": git(args.metadata, "remote", "get-url", "origin").stdout.decode().strip(),
        "defects4j_on_path": bool(shutil.which("defects4j")), "java_on_path": bool(shutil.which("java")),
        "real_input_gate": "blocked_pending_source_failure_replay_candidates_and_gold", "defects": rows,
    })
    return 3


def gguf_tokenizer_metadata(path, *, retain_vocab=False):
    """Read the bounded GGUF v3 metadata section only, leaving tensors untouched."""
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
               10: "Q", 11: "q", 12: "d"}
    vocabulary_arrays = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type"}
    with path.open("rb") as stream:
        ceiling = min(path.stat().st_size, 32 * 1024 * 1024)

        def read(length):
            require(0 <= length <= ceiling - stream.tell(), "GGUF metadata exceeds offline read bound")
            value = stream.read(length)
            require(len(value) == length, "Incomplete GGUF metadata")
            return value

        def scalar(code):
            return struct.unpack("<" + code, read(struct.calcsize("<" + code)))[0]

        def string():
            return read(scalar("Q"))

        require(read(4) == b"GGUF" and scalar("I") == 3, "Expected local little-endian GGUF v3")
        tensor_count, fields = scalar("Q"), scalar("Q")
        require(fields <= 4096, "GGUF metadata has too many fields")
        retained = {"tensor_count_declared": tensor_count}
        for _ in range(fields):
            key = string().decode("utf-8")
            tag = scalar("I")
            if tag == 9:
                elem, count = scalar("I"), scalar("Q")
                require(count <= 1_000_000 and (elem == 8 or elem in formats), "Unsupported GGUF metadata array")
                if retain_vocab and key in vocabulary_arrays:
                    retained[key] = ([string().decode("utf-8") for _ in range(count)] if elem == 8 else
                                     list(struct.unpack(f"<{count}{formats[elem]}",
                                                        read(count * struct.calcsize("<" + formats[elem])))))
                elif elem == 8:
                    for _ in range(count):
                        string()
                else:
                    read(count * struct.calcsize("<" + formats[elem]))
                if key.startswith("tokenizer."):
                    retained[key + ".entries"] = count
            elif tag == 8:
                value = string()
                if key in {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.chat_template"}:
                    retained[key] = value.decode("utf-8")
            else:
                require(tag in formats, "Unknown GGUF metadata type")
                value = scalar(formats[tag])
                if key.startswith("tokenizer.ggml."):
                    retained[key] = value
        retained["metadata_bytes_read"] = stream.tell()
        return retained


def serving(args):
    manifest = load_json(args.manifest)
    require(file_hash(args.manifest) == args.digest, "Installed model manifest digest mismatch")

    def layer(suffix):
        item, = [row for row in manifest["layers"] if row["mediaType"].endswith("." + suffix)]
        digest = item["digest"].removeprefix("sha256:")
        require(SHA256.fullmatch(digest) is not None, "Invalid layer digest")
        path = args.models_dir / "blobs" / ("sha256-" + digest)
        require(path.is_file() and path.stat().st_size == item["size"], f"Missing or incomplete {suffix} blob")
        return digest, path

    template_hash, template = layer("template")
    require(file_hash(template) == template_hash, "Installed serving template bytes differ")
    model_hash, model = layer("model")
    metadata = gguf_tokenizer_metadata(model)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "serving-template.gotmpl").write_bytes(template.read_bytes())
    write_json(args.output / "serving-readiness.json", {
        "schema": "graft-local-serving-audit-v1", "model_manifest_sha256": args.digest,
        "ollama_template_sha256": template_hash, "declared_model_blob_sha256": model_hash,
        "weight_tensor_bytes_read": 0, "gguf_tokenizer_metadata": metadata,
        "serving_token_parity": "blocked_no_trusted_serving_token_id_reference",
        "provider_reported_usage": None, "provider_requests": 0,
    })
    return 3


def compare(args):
    audit, reference, plan = (load_json(path) for path in (args.audit, args.reference, args.plan))
    require(reference["model_manifest_sha256"] == audit["model_manifest_sha256"]
            == plan["config"]["benchmark"]["model_digest"]
            and reference["ollama_template_sha256"] == audit["ollama_template_sha256"],
            "Token reference model or template identity mismatch")
    require(reference.get("serving_build") and reference.get("capture_kind") in {
        "offline_serving_tokenizer", "archived_server_token_ids",
    }, "Reference needs serving build and explicit token capture provenance")
    payloads = {}
    for pair in plan["pairs"]:
        require(pair["status"] == "ready", "Token parity requires fully prepared pairs")
        for rows in pair["arms"].values():
            for row in rows:
                request = row["request"]
                payloads[fingerprint(request)] = request["messages"]
    by_hash = {r["request_sha256"]: r for r in reference["requests"]}
    require(payloads and len(by_hash) == len(reference["requests"]) and set(by_hash) == set(payloads),
            "Reference must cover every distinct planned request exactly")
    counter = TokenizerCounter(args.tokenizer)
    rows = []
    for identity, messages in sorted(payloads.items()):
        entry = by_hash[identity]
        ids = entry["token_ids"]
        require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids),
                "Reference must contain exact token IDs")
        local = counter.token_ids(messages)
        reported = entry.get("provider_prompt_eval_count")
        require(reported is None or type(reported) is int and reported >= 0, "Invalid provider usage field")
        rows.append({
            "request_sha256": identity, "local_template_tokens": len(local),
            "byte_proxy_units": ByteCounter().count(messages),
            "reference_token_ids_count": len(ids), "identical_ids": ids == local,
            "provider_prompt_eval_count": reported,
        })
    matched = all(r["identical_ids"] for r in rows)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "reference-comparison.json", {
        "local_counter": counter.name, "reference_sha256": file_hash(args.reference),
        "serving_build_declared_by_reference": reference["serving_build"],
        "status": "provided_reference_matches" if matched else "mismatch",
        "provider_requests": 0, "admission_bound_changed": False, "requests": rows,
    })
    return 0 if matched else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("defects4j")
    for name in ("metadata", "flexfl", "checkouts"):
        a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--revision", required=True)
    a = sub.add_parser("serving")
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--models-dir", type=Path, required=True)
    a.add_argument("--digest", required=True)
    a = sub.add_parser("compare")
    for name in ("audit", "plan", "reference", "tokenizer"):
        a.add_argument("--" + name, type=Path, required=True)
    for a in sub.choices.values():
        a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return {"defects4j": defects4j, "serving": serving, "compare": compare}[args.command](args)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"Phase 6 audit: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
