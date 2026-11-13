"""One offline launcher for the fixed T0/T1 experiment on macOS and Linux."""
import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
from urllib.parse import urlsplit

from rave_fl.eval.adapters import DATASETS, load_instances, read_rows
from rave_fl.eval.metrics import load_gold
from rave_fl.eval.schema import CompatibilityError, file_hash, fingerprint, safe_name
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.evaluation import PROFILES, evaluate_profile
from rave_fl.graft.fixed_evidence_reporting import audit_saved_run, load_json, report, require, write_report
from rave_fl.graft.prompt_settlement import SettlementCertificate
from rave_fl.graft.ollama_live import LiveOllamaPaths
from rave_fl.pilot import write_json


PROTOCOL = "graft-fixed-t0-t1-v1"
LOCK_SCHEMA = "graft-fixed-cohort-v1"
HOSTS = {"local_mac": "Darwin", "remote_linux_a100": "Linux"}
PROFILE_ROOT = Path(__file__).resolve().parents[2] / "configs/graft_portable"
BENCHMARK_FLAGS = (
    "model", "model_digest", "backend", "endpoint", "dataset", "dataset_file", "checkouts_dir",
    "candidate_file", "seed", "sample", "device", "budget", "context_window", "output_tokens",
    "physical_budget", "per_bug_physical_budget", "max_new_calls",
)
CONFIG_FLAGS = (
    "embeddings_dir", "tokenizer_dir", "labels_file", "input_allowance",
    "per_arm_physical_budget", "per_arm_max_new_calls", "per_bug_max_new_calls",
)
# Paths and host placement can vary. Exact per-target requests and resource content are bound separately.
OPERATIONAL = {
    "dataset_file", "checkouts_dir", "repo_cache", "candidate_file", "split_manifest",
    "endpoint", "device", "timeout", "output_dir", "bugs", "limit", "sample",
}


def read_host(name, simulate, backend):
    path = Path(name) if name not in HOSTS else PROFILE_ROOT / (name + ".json")
    data = load_json(path)
    require(set(data) == {"schema", "name", "system", "benchmark"}
            and data["schema"] == "graft-host-profile-v1"
            and HOSTS.get(data["name"]) == data["system"]
            and isinstance(data["benchmark"], dict)
            and set(data["benchmark"]) <= {"device", "endpoint", "timeout"},
            "Invalid host profile or an attempt to set scientific options in the host profile")
    require(not simulate or backend == "fake", "Host simulation is available only with the fake backend")
    require(platform.system() == data["system"] or simulate, "Host profile does not match this operating system")
    return data


def resolve_config(args):
    value = load_json(args.config)
    benchmark = value.setdefault("benchmark", {})
    backend = args.backend if args.backend is not None else benchmark.get("backend", "ollama")
    host = read_host(args.profile, args.simulate_host, backend)
    benchmark.update(host["benchmark"])
    for name in BENCHMARK_FLAGS:
        if getattr(args, name) is not None:
            benchmark[name] = getattr(args, name)
    for name in CONFIG_FLAGS:
        if getattr(args, name) is not None:
            value[name] = getattr(args, name)
    if args.bugs is not None or args.bug_file is not None:
        raw = args.bugs.split(",") if args.bugs is not None else args.bug_file.read_text().splitlines()
        benchmark["bugs"] = [item.strip() for item in raw]
        require(all(benchmark["bugs"]), "Bug IDs must be nonempty")
    if args.all:
        benchmark.update(bugs=[], limit=None)
    if args.limit is not None:
        benchmark["limit"] = args.limit
    ids = benchmark.get("bugs", [])
    require(isinstance(ids, list) and all(isinstance(i, str) and i.strip() for i in ids)
            and len(ids) == len(set(ids)), "Duplicate or malformed requested bug IDs")
    # Use the existing fixed runner validator, including the no-download and strict-policy rules.
    config = fixed.configuration(value)
    endpoint = urlsplit(config["benchmark"]["endpoint"])
    require(endpoint.scheme in {"http", "https"} and endpoint.hostname and not endpoint.username
            and not endpoint.password and not endpoint.query and not endpoint.fragment,
            "Use an explicit HTTP(S) endpoint without credentials, query, or fragment")
    require(config["benchmark"]["backend"] == "fake"
            or host["name"] != "local_mac" or config["benchmark"]["device"] not in {"cuda", "a100"},
            "CUDA/A100 device selection is incompatible with the Mac profile")
    return config, host


def select(config):
    opts = config["benchmark"]
    # Validate explicit IDs before limiting the selected cohort. The adapter supplies the canonical order.
    instances, source = load_instances({**opts, "limit": None})
    require(set(opts["bugs"]) <= {i.instance_id for i in instances},
            "Some explicit bug IDs were excluded by dataset or split filters")
    instances = instances[:opts["limit"]]
    require(instances and len({i.provenance["independent_bug_id"] for i in instances}) == len(instances),
            "Empty selection or duplicate independent defects")
    return instances, source


def partition(ids, count):
    require(type(count) is int and 1 <= count <= len(ids) and len(ids) == len(set(ids)),
            "Shard count must be positive and no greater than the unique cohort size")
    return [sorted(ids[index::count]) for index in range(count)]


def shard_config(config, ids):
    return {**config, "benchmark": {**config["benchmark"], "bugs": ids, "limit": None, "sample": "first"}}


def label_identity(config):
    return file_hash(config["labels_file"]) if config["labels_file"] else None


def profile_for(config):
    return "java_strict_v2" if config["benchmark"]["dataset"] == "defects4j" else "legacy_v1"


def condition(plan, labels, split_sha256):
    config = plan["config"]
    opts = config["benchmark"]
    root = Path(config["tokenizer_dir"]) if config["tokenizer_dir"] else None
    tokenizer = {str(Path(p).relative_to(root)): h for p, h in plan["resources"].items()
                 if root and Path(p).is_relative_to(root)}
    return {
        "protocol": PROTOCOL,
        "dataset_sha256": plan["dataset"]["dataset_sha256"],
        "dataset_release": plan["dataset"]["dataset_release"],
        "split_manifest_sha256": split_sha256,
        "labels_sha256": labels, "evaluation_profile": profile_for(config),
        "benchmark": {k: v for k, v in opts.items() if k not in OPERATIONAL},
        "configuration": {k: v for k, v in config.items() if k not in {
            "benchmark", "embeddings_dir", "tokenizer_dir", "labels_file", "prompt_settlement",
        }},
        **({"execution_admission": plan["settlement_serving_identity"]} if "settlement_serving_identity" in plan else {}),
        "tokenizer_files": tokenizer, "has_local_tokenizer": root is not None,
        "arm_configs": plan["arm_configs"], "arm_diff": plan["arm_diff"],
        "system_prompt_sha256": plan["system_prompt_sha256"], "implementation": plan["implementation"],
    }


def shard_record(plan):
    return {
        "ids": [pair["instance_id"] for pair in plan["pairs"]],
        # The raw preparation remains bound by the local resume manifest.
        # Portable preparation and exact requests remain bound across hosts.
        "pair_fingerprints": {
            pair["instance_id"]: fingerprint({k: v for k, v in pair.items() if k != "prepared_sha256"})
            for pair in plan["pairs"]
        },
        "planned_calls": plan["planned_fresh_calls_upper"],
        "planned_physical_tokens": plan["physical_reservation_upper"],
        "admitted": plan["run_admitted"],
    }


def split_identity(config):
    path = config["benchmark"].get("split_manifest")
    return file_hash(path) if path else None


def read_lock(path):
    lock = load_json(path)
    digest = lock.pop("sha256")
    require(lock["schema"] == LOCK_SCHEMA and fingerprint(lock) == digest, "Cohort lock checksum mismatch")
    groups = partition(lock["ids"], len(lock["shards"]))
    require([row["ids"] for row in lock["shards"]] == groups, "Cohort shard coverage or order differs")
    for group, shard in zip(groups, lock["shards"], strict=True):
        require(set(shard["pair_fingerprints"]) == set(group), "Missing or foreign defect fingerprint")
    require(sum(shard["planned_calls"] for shard in lock["shards"]) <= lock["max_new_calls"]
            and sum(shard["planned_physical_tokens"] for shard in lock["shards"]) <= lock["physical_budget"],
            "Cohort lock exceeds aggregate caps")
    return {**lock, "sha256": digest}


def binding(lock, index):
    return {"cohort_sha256": lock["sha256"], "shard_index": index,
            "labels_sha256": lock["condition"]["labels_sha256"],
            "split_manifest_sha256": lock["condition"]["split_manifest_sha256"],
            **({"live_attestation": lock["live_attestation"]} if "live_attestation" in lock else {})}


def validate_shard(lock, index, plan, labels, split):
    require(type(index) is int and 0 <= index < len(lock["shards"]), "Shard index is outside the cohort")
    require(condition(plan, labels, split) == lock["condition"],
            "Scientific condition, model, dataset, tokenizer, labels or implementation differs from cohort")
    require(shard_record(plan) == lock["shards"][index],
            "Shard differs from its frozen candidate pools, evidence, prompts, reservations or membership")


def optional_model_check(args, opts):
    details = {"identity": opts["model_digest"], "status": "simulated" if opts["backend"] == "fake"
               else "unverified"}
    if args.generation_manifest:
        path = args.generation_manifest.resolve()
        details["manifest_sha256"] = file_hash(path)
        manifest = load_json(path)
        # An explicit Ollama root or the standard parent of its manifests tree.
        root = args.ollama_models_dir
        require(root is not None, "Use --ollama-models-dir with --generation-manifest")
        layers = manifest["layers"] + [manifest["config"]]
        for layer in layers:
            digest = layer["digest"]
            require(digest.startswith("sha256:") and len(digest) == 71
                    and all(c in "0123456789abcdef" for c in digest[7:]), "Malformed model layer digest")
            blob = root / "blobs" / digest.replace(":", "-")
            require(blob.is_file() and blob.stat().st_size == layer["size"], "Missing or incomplete model blob: " + str(blob))
        if opts["model_digest"]:
            require(details["manifest_sha256"] == opts["model_digest"].removeprefix("sha256:"),
                    "Local generation manifest does not match the configured digest")
        details["status"] = "local_manifest_and_blob_sizes_verified"
    if args.check_provider:
        require(opts["backend"] in {"fake", "ollama"}, "Metadata checks support fake and Ollama only")
        # Provider.preflight only reads model metadata. The runner is still offline-only for real backends.
        from rave_fl.eval.backend import Provider
        details["provider_metadata"] = Provider(opts).preflight()
        details["status"] = "metadata_verified"
    return details


def inspect_encoder(path):
    from rave_fl.graft.topology import has_bge_m3_files
    require(path.is_dir(), "Local BGE-M3 checkpoint directory is missing")
    require(has_bge_m3_files(p.name for p in path.iterdir() if p.is_file() and p.stat().st_size),
            "Local BGE-M3 directory lacks weights, heads, configuration or tokenizer files")
    require(all(find_spec(name) for name in ("torch", "transformers", "FlagEmbedding")),
            "Local encoder needs torch, transformers and FlagEmbedding")


def label_inventory(config, instances, pairs):
    if not config["labels_file"]:
        return {i.instance_id: {"status": "missing_labels"} for i in instances}
    path = Path(config["labels_file"])
    rows, _ = read_rows(path)
    ids = [row.get("instance_id") or row.get("bug") for row in rows]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids))
            and all("gold" in row and "gold_file" not in row for row in rows),
            "Offline labels must have unique IDs and inline gold")
    present = set(ids)
    output = {}
    for instance in instances:
        ident = instance.instance_id
        if ident not in present:
            output[ident] = {"status": "missing_labels"}
            continue
        gold = load_gold(instance, {"dataset_file": str(path), "path_base": str(path.parent)})
        if not gold.functions or ident not in pairs:
            output[ident] = {"status": "missing_method_labels" if not gold.functions else "pool_unavailable"}
            continue
        pair = pairs[ident]
        candidates = [candidate.function_id for candidate in pair.schedule]
        metric = evaluate_profile({"ranked_functions": candidates}, pair.entities, gold,
                                  instance.dataset, profile_for(config))["method"]
        recall = metric["candidate_pool_coverage"] if metric else None
        output[ident] = {
            "status": "method_labels_unresolved" if recall is None else
            "full_pool_recall" if recall == 1 else "candidate_pool_miss" if recall == 0 else "partial_pool_recall",
            "candidate_recall": recall,
        }
    return output


def preflight(args, config, host):
    args.output.mkdir(parents=True, exist_ok=False)
    diagnostics = {
        "status": "blocked", "new_model_calls": 0, "host_profile": host["name"],
        "simulation": args.simulate_host, "system": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "free_output_bytes": shutil.disk_usage(args.output).free,
        "tools": {name: bool(shutil.which(name)) for name in ("git", "java", "defects4j", "nvidia-smi")},
        "dependencies": {name: bool(find_spec(name)) for name in
                         ("tree_sitter", "tree_sitter_java", "transformers", "FlagEmbedding", "torch", "matplotlib", "numpy")},
        "real_execution": "disabled", "serving_template_parity": "unverified",
        "errors": [], "inventory": [],
    }
    try:
        opts = config["benchmark"]
        live_attestation = None
        attested_provider = None
        if args.attest_live:
            require(config.get("prompt_settlement") and not args.simulate_host,
                    "Live attestation requires an audited settlement config on the actual host")
            require(args.shard_count == 1 and args.cohort_max_new_calls is None and args.cohort_physical_budget is None,
                    "Live smoke requires one cohort with the original caps")
            attested_provider = live_paths(args).provider(config)
            live_attestation = attested_provider.preflight()
            diagnostics["live_attestation"] = live_attestation
            diagnostics["real_execution"] = "manual_only"
        if config.get("prompt_settlement"):
            require(args.shard_count == 1 and not args.check_provider, "Audited smoke requires one offline shard")
            if not live_attestation:
                certificate = SettlementCertificate(config)
                certificate.verify_local_host(args.serving_binary, args.generation_manifest, args.ollama_models_dir,
                                              platform.system(), platform.machine())
            diagnostics["serving_template_parity"] = "verified_offline_for_audited_requests"
        instances, source = select(config)
        ids = [i.instance_id for i in instances]
        groups = partition(ids, args.shard_count)
        require(args.shard_count == 1 or (args.cohort_physical_budget and args.cohort_max_new_calls),
                "Sharding requires explicit whole-cohort token and call caps")
        physical_cap = args.cohort_physical_budget or opts["physical_budget"]
        call_cap = args.cohort_max_new_calls or opts["max_new_calls"]
        diagnostics["model"] = optional_model_check(args, opts)
        if args.encoder_model_dir:
            inspect_encoder(args.encoder_model_dir)
            diagnostics["encoder_checkpoint_layout"] = "present"
        diagnostics["embeddings_dir_exists"] = Path(config["embeddings_dir"]).is_dir()
        diagnostics["local_tokenizer_configured"] = bool(config["tokenizer_dir"])
        require(diagnostics["tools"]["git"], "Git is required for checkout validation")
        if any(i.language == "java" for i in instances):
            require(diagnostics["dependencies"]["tree_sitter"] and diagnostics["dependencies"]["tree_sitter_java"],
                    "Java preparation requires tree-sitter and tree-sitter-java")
        require(diagnostics["embeddings_dir_exists"], "Embedding directory missing. Prepare local embeddings explicitly.")
        labels = label_identity(config)
        split = split_identity(config)
        by_id = {i.instance_id: i for i in instances}
        records, signatures, label_status = {}, [], {}
        first_condition = None
        for index, group in enumerate(groups):
            plan, pairs = fixed.prepare(shard_config(config, group))
            signature = shard_record(plan)
            require(signature["ids"] == group, "Prepared shard selection drifted")
            current = condition(plan, labels, split)
            require(current["dataset_sha256"] == source["dataset_sha256"], "Dataset changed during preflight")
            if first_condition is None:
                first_condition = current
            require(current == first_condition, "Resources or scientific condition changed during preflight")
            signatures.append(signature)
            label_status.update(label_inventory(config, [by_id[ident] for ident in group], pairs))
            records.update({p["instance_id"]: p for p in plan["pairs"]})
            write_json(args.output / "plans" / f"shard-{index:03d}.json", plan)
        require(label_identity(config) == labels and split_identity(config) == split,
                "Labels or split changed during preflight")
        diagnostics["inventory"] = [{
            "instance_id": ident, "revision": by_id[ident].revision, "checkout": by_id[ident].checkout,
            "evidence_sha256": by_id[ident].evidence.sha256, "evidence_kind": by_id[ident].evidence.kind,
            "evidence_source": by_id[ident].evidence.source, "split": opts.get("split"),
            "pool_sha256": records[ident].get("pool_sha256"), "status": records[ident]["status"],
            "reasons": records[ident]["reasons"], "scheduled_candidates": len(records[ident]["schedule"]),
            "offline_labels": label_status[ident],
            **({"arm_admission": {name: {
                "status": "PASS" if bound["serial_peak_tokens"] <= min(opts["budget"], config["per_arm_physical_budget"]) else "FAIL",
                **bound,
                "settled_upper_tokens": sum(row["settled_upper_tokens"] for row in records[ident]["arms"][name]),
                "provider_reported_tokens": None,
            } for name, bound in records[ident]["physical_envelope_by_arm"].items()},
                "pair_admission": records[ident]["physical_envelope"],
            } if config.get("prompt_settlement") and records[ident].get("physical_envelope_by_arm") else {}),
        } for ident in ids]
        calls = sum(s["planned_calls"] for s in signatures)
        reserve = sum(s["planned_physical_tokens"] for s in signatures)
        diagnostics.update(selected=len(ids), planned_calls=calls, conservative_cohort_tokens=reserve)
        require(calls <= call_cap and reserve <= physical_cap, "Combined shard reservations exceed the explicit cohort caps")
        lock = {"schema": LOCK_SCHEMA, "ids": ids, "condition": first_condition, "shards": signatures,
                "physical_budget": physical_cap, "max_new_calls": call_cap}
        if live_attestation:
            attested_provider.before_fresh_request()
            require(all(record["status"] == "ready" for record in records.values())
                    and signatures[0]["admitted"] and calls == sum(2 * len(record["schedule"]) for record in records.values()),
                    "Live cohort requires every planned request to be admissible")
            lock["live_attestation"] = live_attestation
        write_json(args.output / "cohort.json", {**lock, "sha256": fingerprint(lock)})
        if all(record["status"] == "ready" for record in records.values()):
            diagnostics["status"] = "ready_for_manual_live" if live_attestation else "ready_for_offline_plan"
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError, subprocess.SubprocessError) as exc:
        diagnostics["errors"].append(str(exc))
    write_json(args.output / "preflight.json", diagnostics)
    print(json.dumps({"status": diagnostics["status"], "output": str(args.output), "new_model_calls": 0}))
    return 0 if diagnostics["status"] in {"ready_for_offline_plan", "ready_for_manual_live"} else 3


def live_paths(args):
    return LiveOllamaPaths(args.serving_binary, args.generation_manifest, args.ollama_models_dir)


def collect(args, config):
    live = config["benchmark"]["backend"] != "fake"
    if live:
        require(args.allow_inference and config.get("prompt_settlement") and config["benchmark"]["backend"] == "ollama",
                "Real inference requires --allow-inference and a certified Ollama configuration")
    require(not (args.bugs or args.bug_file or args.all or args.limit or args.sample),
            "Freeze bug selection and sampling in preflight. Run/resume uses the cohort membership.")
    lock = read_lock(args.cohort)
    if live:
        certificate = SettlementCertificate(config)
        certificate.verify_live_caps(config)
        require(tuple(lock["ids"]) == certificate.cohort == tuple(config["benchmark"]["bugs"])
                and len(lock["shards"]) == 1 and lock["shards"][0]["admitted"]
                and lock["shards"][0]["planned_calls"] == certificate.calls
                and lock["physical_budget"] == config["benchmark"]["physical_budget"]
                and lock["max_new_calls"] == config["benchmark"]["max_new_calls"]
                and lock.get("live_attestation", {}).get("prompt_settlement_identity") == certificate.identity,
                "Run requires a live-attested cohort for this exact certified schedule")
    index = args.shard_index
    require(0 <= index < len(lock["shards"]), "Shard index is outside the cohort")
    config = shard_config(config, lock["shards"][index]["ids"])
    prepared = fixed.prepare(config)
    plan, _ = prepared
    validate_shard(lock, index, plan, label_identity(config), split_identity(config))
    result = fixed.run(config, args.output, resume=args.command == "resume", allow_inference=args.allow_inference,
                       cohort_binding=binding(lock, index), expected_plan_sha256=fingerprint(plan),
                       live_paths=live_paths(args) if live else None, prepared=prepared)
    calls = result["new_calls_this_invocation"]
    print(json.dumps({"output": str(args.output), "new_simulated_calls": 0 if live else calls,
                      "real_llm_calls": calls if live else 0, "spending": result["spending"]}))
    return 0 if all(p["pair_status"] == "complete" for p in result["pairs"]) else 3


def merge(lock, runs, output):
    output = Path(output).resolve()
    require(all(not output.is_relative_to(path.resolve()) for path in runs),
            "Choose an aggregate output outside every input run")
    shards, keys = {}, set()
    for path in runs:
        rows, evidence, profile, manifest, spending, labels = audit_saved_run(path, None)
        attached = manifest.get("cohort_binding")
        require(isinstance(attached, dict), "Run has no portable cohort binding")
        index = attached["shard_index"]
        require(attached == binding(lock, index) and index not in shards,
                "Foreign cohort binding or duplicate shard index")
        validate_shard(lock, index, manifest["plan"], labels, attached["split_manifest_sha256"])
        result = load_json(path / "results.json")
        request_keys = {call["request_key"] for pair in result["pairs"]
                        for arm in pair["arms"].values() for call in arm["calls"]}
        require(not (keys & request_keys), "Duplicate paid request key across shards")
        keys.update(request_keys)
        shards[index] = rows, evidence, profile, manifest, spending, path
    require(set(shards) == set(range(len(lock["shards"]))), "Incomplete shard coverage")
    first = shards[0]
    require(all(entry[2] == first[2] and entry[3]["provider"] == first[3]["provider"]
                for entry in shards.values()), "Provider identity or evaluation profile differs")
    order = {ident: i for i, ident in enumerate(lock["ids"])}
    rows = sorted((row for entry in shards.values() for row in entry[0]),
                  key=lambda row: (order[row["instance_id"]], row["arm"]))
    require(len(rows) == 2 * len(order) and len({(r["instance_id"], r["arm"]) for r in rows}) == len(rows),
            "Duplicate or missing defect arms across shards")
    evidence = sorted((row for entry in shards.values() for row in entry[1]),
                      key=lambda row: (order[row["instance_id"]], row["target"]))
    spending = dict(first[4])
    for key, value in spending.items():
        if type(value) is int:
            spending[key] = sum(entry[4][key] for entry in shards.values())
    require(spending["new_calls"] <= lock["max_new_calls"]
            and spending["known_provider_tokens"] <= lock["physical_budget"], "Observed spending exceeds cohort caps")
    hashes = {f"shard-{index:03d}/{name}": file_hash(shards[index][5] / name)
              for index in sorted(shards) for name in ("manifest.json", "results.json", "accounting.json")}
    plans = [shards[index][3]["plan"] for index in sorted(shards)]
    plan = {**plans[0], "pairs": [pair for p in plans for pair in p["pairs"]],
            "physical_envelope": {
                "sum_independent_shard_serial_peaks_tokens": sum(p["physical_reservation_upper"] for p in plans),
                "sum_full_window_reservations_tokens": sum(
                    p["physical_envelope"]["sum_full_window_reservations_tokens"] for p in plans
                ),
            }}
    return write_report(output, rows, evidence, first[2], plan, spending,
                        lock["condition"]["labels_sha256"], hashes, cohort={
                            "cohort_sha256": lock["sha256"], "shard_count": len(shards),
                            "cohort_physical_budget": lock["physical_budget"], "cohort_max_new_calls": lock["max_new_calls"],
                        })


def prepare_embeddings(args, config):
    # This explicit encoder-only command reuses the frozen graph preparation.
    from rave_fl.eval.runner import prepare_one
    from rave_fl.graft.encode import export
    from rave_fl.graft.fixed_inputs import fixed_graph
    instances, _ = select(config)
    directory = Path(config["embeddings_dir"])
    inspect_encoder(args.model_dir)
    groups = partition([i.instance_id for i in instances], args.shard_count)
    require(0 <= args.shard_index < len(groups), "Shard index is outside the selected inputs")
    selected = set(groups[args.shard_index])
    rows, metadata = read_rows(config["benchmark"]["dataset_file"])
    require(not any({"gold", "gold_file", "patch", "fix_patch"} & row.keys() for row in [metadata, *rows]),
            "Online embedding inputs must exclude labels and patches")
    prepared = []
    for instance in instances:
        if instance.instance_id not in selected:
            continue
        require(not instance.provenance.get("skip_reason"), f"Excluded defect {instance.instance_id}")
        instance, entities, candidates, *_ = prepare_one(instance, config["benchmark"])
        fixed.fixed_schedule(candidates, entities)
        target = directory / (safe_name(instance.instance_id) + ".json")
        require(not target.exists(), f"Embedding output already exists: {target}")
        prepared.append((fixed_graph(instance, entities), target))
    for graph, target in prepared:
        export(graph, args.model_dir, args.revision, target, device="cpu")
        print(target)
    return 0


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def common(parser):
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True, help="Built-in host profile or local profile JSON")
    parser.add_argument("--simulate-host", action="store_true", help="Permit an OS mismatch with the fake backend")
    for key in BENCHMARK_FLAGS + CONFIG_FLAGS:
        kwargs = {}
        if key in {"backend", "dataset", "sample", "device"}:
            kwargs["choices"] = {"backend": ["fake", "ollama", "plugin"], "dataset": list(DATASETS),
                                 "sample": ["first", "random"], "device": ["auto", "cpu", "apple", "cuda", "a100"]}[key]
        if key == "seed":
            kwargs["type"] = int
        if key in {"budget", "context_window", "output_tokens", "physical_budget", "per_bug_physical_budget",
                   "max_new_calls", "input_allowance", "per_arm_physical_budget", "per_arm_max_new_calls",
                   "per_bug_max_new_calls"}:
            kwargs["type"] = positive
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--bugs", help="Comma-separated bug IDs")
    selection.add_argument("--bug-file", type=Path, help="One bug ID per line")
    selection.add_argument("--all", action="store_true", help="Clear configured bug IDs and sample count")
    parser.add_argument("--limit", type=positive, help="Sample count applied before sharding")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List supported adapters and built-in profiles")
    for name in ("preflight", "dry-run"):
        p = sub.add_parser(name, help="Write diagnostics and freeze a whole cohort without model calls")
        common(p)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--shard-count", type=positive, default=1)
        p.add_argument("--cohort-physical-budget", type=positive)
        p.add_argument("--cohort-max-new-calls", type=positive)
        p.add_argument("--check-provider", action="store_true", help="Read provider metadata only")
        p.add_argument("--attest-live", action="store_true", help="Seal local listener and model identity using metadata only")
        p.add_argument("--generation-manifest", type=Path, help="Explicit local Ollama model manifest")
        p.add_argument("--serving-binary", type=Path, help="Local binary verified by the settlement audit")
        p.add_argument("--ollama-models-dir", type=Path)
        p.add_argument("--encoder-model-dir", type=Path, help="Check a local BGE-M3 layout without loading it")
    for name in ("run", "resume"):
        p = sub.add_parser(name, help="Execute or resume a frozen cohort shard")
        common(p)
        p.add_argument("--cohort", required=True, type=Path)
        p.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index")
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--allow-inference", action="store_true", help="Explicitly authorize generation for an attested cohort")
        p.add_argument("--serving-binary", type=Path)
        p.add_argument("--generation-manifest", type=Path)
        p.add_argument("--ollama-models-dir", type=Path)
    p = sub.add_parser("prepare-embeddings", help="Explicit CPU encoding with an existing local checkpoint")
    common(p)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--shard-count", type=positive, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    for name in ("report", "plot"):
        p = sub.add_parser(name, help="Regenerate tables and PNG/SVG figures from a sealed run")
        p.add_argument("--run", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--evaluation-profile", choices=PROFILES)
    p = sub.add_parser("aggregate", help="Audit and report every sealed shard in a frozen cohort")
    p.add_argument("--cohort", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path, action="append")
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print(json.dumps({"datasets": list(DATASETS), "profiles": HOSTS,
                              "fixed_runner_languages": ["java", "python"], "real_execution": "manual_attested_ollama"}, indent=2))
            return 0
        if args.command in {"report", "plot"}:
            report(args.run, args.output, args.evaluation_profile)
            return 0
        if args.command == "aggregate":
            merge(read_lock(args.cohort), args.run, args.output)
            return 0
        config, host = resolve_config(args)
        if args.command in {"preflight", "dry-run"}:
            return preflight(args, config, host)
        if args.command == "prepare-embeddings":
            return prepare_embeddings(args, config)
        return collect(args, config)
    except (ValueError, OSError, ImportError, KeyError, TypeError, sqlite3.Error, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"GRAFT experiment: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
