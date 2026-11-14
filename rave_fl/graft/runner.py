"""Opt-in GRAFT execution using existing dataset, candidate, provider and rankers."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from rave_fl.eval.cli import configuration
from rave_fl.eval.adapters import load_instances
from rave_fl.eval.backend import Backend, Provider, Store, ExecutionLimit, UnknownUsage
from rave_fl.eval.fusion import Fusion
from rave_fl.eval.metrics import load_gold
from rave_fl.eval.policies import engine
from rave_fl.eval.runner import prepare_one, run_lock, implementation_identity
from rave_fl.eval.schema import (
    OnlineInstance,
    CodeEntity,
    CompatibilityError,
    fingerprint,
    file_hash,
    safe_name,
)
from rave_fl.pilot import write_json
from rave_fl.graft.controller import GraftConfig, Session, Router, GraftProvider
from rave_fl.graft.graph import build_graph
from rave_fl.graft.selection import ByteCounter, TokenizerCounter
from rave_fl.graft.topology import EmbeddingBundle
from rave_fl.graft.evaluation import evaluate_profile, PROFILES

BASELINES = {"fixed_l1", "highest_suspicion", "v2_utility", "full"}
GRAFT_POLICIES = {"l1_anchor", "l1_graph", "l1_graph_suspicion", "l1_graph_focused", "l1_graph_test_aware", "graft_adaptive"}


def validate_config(value):
    unknown = set(value) - {
        "benchmark",
        "graft",
        "graft_policies",
        "embeddings_dir",
        "tokenizer_dir",
    }
    if unknown:
        raise CompatibilityError("Unknown GRAFT config fields: " + str(sorted(unknown)))
    benchmark = dict(value.get("benchmark", {}))
    from rave_fl.eval.cli import DEFAULTS

    if set(benchmark) - set(DEFAULTS):
        raise CompatibilityError("Unknown benchmark settings")
    opts = configuration(
        Namespace(
            **{
                "admission": "strict",
                "budget": 24000,
                "batch_size": 2,
                "context_window": 8192,
                "policies": ["fixed_l1", "highest_suspicion", "v2_utility", "full"],
                "fusions": ["rrf"],
                **benchmark,
            }
        )
    )
    config = GraftConfig(**value.get("graft", {}))
    policies = value.get("graft_policies", ["l1_graph", "graft_adaptive"])
    if (
        not isinstance(policies, list)
        or len(set(policies)) != len(policies)
        or set(policies) - GRAFT_POLICIES
    ):
        raise CompatibilityError("Unknown or repeated GRAFT policies")
    if set(opts["policies"]) - BASELINES:
        raise CompatibilityError(
            "GRAFT wrapper supports fixed_l1, highest_suspicion, v2_utility and full baselines"
        )
    if (
        opts["admission"] != "strict"
        or opts["granularity"] != "method"
        or opts["representation"] != "l1"
    ):
        raise CompatibilityError(
            "GRAFT needs strict admission, method candidates and L1 representations"
        )
    if (
        opts["workers"] != 1
        or opts["evaluation_mode"] != "policy"
        or opts.get("replay_run")
        or opts.get("observations_dir")
    ):
        raise CompatibilityError(
            "Use serial live/fake collection here; historical replay remains in the existing RAVE CLI"
        )
    if opts.get("allow_downloads") or opts.get("materialize_checkouts"):
        raise CompatibilityError("Prepare repositories/weights explicitly before GRAFT execution")
    if config.max_full != opts["max_full"]:
        raise CompatibilityError("Use the same max_full for simple and GRAFT policies")
    for key in [
        "dataset_file",
        "candidate_file",
        "checkouts_dir",
        "reservation_spec",
        "cache_dir",
        "split_manifest",
    ]:
        if opts.get(key):
            opts[key] = str(Path(opts[key]).resolve())
    result = {
        "benchmark": opts,
        "graft": asdict(config),
        "graft_policies": policies,
        "embeddings_dir": value.get("embeddings_dir"),
        "tokenizer_dir": value.get("tokenizer_dir"),
    }
    for key in ["embeddings_dir", "tokenizer_dir"]:
        if result[key]:
            result[key] = str(Path(result[key]).resolve())
    if config.selector == "topo" and not result["embeddings_dir"]:
        raise CompatibilityError(
            "Topo selection requires real precomputed embedding bundles; no implicit encoder"
        )
    return result


def resources(config, instance_id=None):
    counter = (
        TokenizerCounter(config["tokenizer_dir"]) if config["tokenizer_dir"] else ByteCounter()
    )
    embeddings = None
    if config["graft"]["selector"] == "topo" and instance_id:
        embeddings = EmbeddingBundle.load(
            Path(config["embeddings_dir"]) / (safe_name(instance_id) + ".json")
        )
        if embeddings.synthetic and config["benchmark"]["backend"] != "fake":
            raise CompatibilityError(
                "Synthetic embedding fixtures cannot be used as real-model evidence"
            )
    return counter, embeddings


def resource_identity(config):
    paths = []
    if config["tokenizer_dir"]:
        # A tokenizer may share a directory with a 200B model. Pin tokenizer
        # metadata/vocabulary/templates without re-reading all model weights.
        paths.extend(
            p
            for p in Path(config["tokenizer_dir"]).rglob("*")
            if p.is_file() and p.suffix in {".json", ".txt", ".model", ".jinja"}
        )
    if config["embeddings_dir"]:
        root = Path(config["embeddings_dir"])
        if not root.is_dir():
            raise CompatibilityError("Embedding bundle directory is missing")
        paths.extend(root.glob("*.json"))
    if config["benchmark"].get("reservation_spec"):
        paths.append(Path(config["benchmark"]["reservation_spec"]))
    return {str(p.resolve()): file_hash(p) for p in sorted(paths)}


def preflight(config):
    from rave_fl.eval.preflight import inspect

    opts = config["benchmark"]
    result = inspect(opts)
    result["graft"] = config["graft"]
    result["graft_policies"] = config["graft_policies"]
    result["package_counter"] = (
        "local model chat template" if config["tokenizer_dir"] else ByteCounter.name
    )
    result["context_is_not_policy_budget"] = True
    result["candidate_preparations"] = []
    try:
        instances, _ = load_instances(opts)
        for instance in instances:
            try:
                i, entities, cs, _, _, _ = prepare_one(instance, opts)
                graph = build_graph(i, entities)
                counter, embeddings = resources(config, i.instance_id)
                if embeddings:
                    from rave_fl.graft.selection import query_text

                    for text in [n.text for n in graph.nodes] + [
                        query_text(graph, c.function_id) for c in cs
                    ]:
                        embeddings.lookup(text)
                result["candidate_preparations"].append(
                    {
                        "instance_id": i.instance_id,
                        "status": "ready",
                        "candidates": len(cs),
                        "graph_nodes": len(graph.nodes),
                        "graph_hash": graph.hash,
                        "counter": counter.name,
                    }
                )
            except Exception as exc:
                result["candidate_preparations"].append(
                    {
                        "instance_id": instance.instance_id,
                        "status": "unavailable",
                        "error": str(exc),
                    }
                )
    except Exception as exc:
        result["preparation_error"] = str(exc)
    result["ready"] = (
        bool(result["candidate_preparations"])
        and all(v["status"] == "ready" for v in result["candidate_preparations"])
        and not result.get("provider_error")
    )
    count = len(result["candidate_preparations"])
    result["graft_call_upper_additional"] = (
        count
        * len(opts["fusions"])
        * len(config["graft_policies"])
        * (config["graft"]["max_graph"] + config["graft"]["max_full"])
    )
    result[
        "expected_cost_note"
    ] = "Hard caps, not an expected bill. L1 batches are exact-cache reusable. Full responses cannot stand in for changed graph prompts. Embedding time/memory is separate from generation tokens."
    return result


def complete(arm, policy, count):
    states = arm.get("states", {})
    if len(states) != count or arm.get("budget_exceeded"):
        return False
    if any(not s["observation"] or not s["observation"]["valid"] for s in states.values()):
        return False
    if policy == "full":
        return all(s["level"] == 2 for s in states.values())
    return all(s["level"] >= 1 for s in states.values()) and not arm.get(
        "stop_reason", ""
    ).startswith("execution_limit")


def run(config, directory, *, resume=False, allow_inference=False, dry_run=False):
    directory = Path(directory).resolve()
    opts = config["benchmark"]
    dry_run = dry_run or opts["dry_run"]
    if opts["backend"] != "fake" and not dry_run and not allow_inference:
        raise CompatibilityError(
            "Real inference is disabled; supply --allow-inference only for the later model experiment"
        )
    if directory.exists() and not resume:
        raise CompatibilityError("Output exists; resume or choose a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        instances, source = load_instances(opts)
        identity = {
            "schema": "graft-run-v1",
            "config": config,
            "dataset_source": source,
            "instances": [i.to_dict() for i in instances],
            "implementation": implementation_identity(),
            "resources": resource_identity(config),
            "dry_run": dry_run,
        }
        identity["sha256"] = fingerprint(identity)
        manifest = directory / "manifest.json"
        if resume:
            if json.loads(manifest.read_text()) != identity:
                raise CompatibilityError(
                    "Resume config, data, resources or implementation changed; create a new run"
                )
            # Validate every saved arm and prepared input before any new paid
            # action, including inputs of bugs scheduled later in this run.
            for instance in instances:
                prior_path = directory / "instances" / safe_name(instance.instance_id)
                if (prior_path / "input.json").exists():
                    prepared = prepare_one(instance, opts)[-1]
                    if fingerprint(prepared) != fingerprint(
                        json.loads((prior_path / "input.json").read_text())
                    ):
                        raise CompatibilityError("Resume prepared buggy inputs changed")
                if (prior_path / "status.json").exists():
                    saved_status = json.loads((prior_path / "status.json").read_text())
                    for entry in saved_status["arms"].values():
                        saved_arm = directory / entry["file"]
                        if not saved_arm.exists() or file_hash(saved_arm) != entry["sha256"]:
                            raise CompatibilityError(
                                "Saved arm checksum mismatch; preserve and investigate"
                            )
        else:
            write_json(manifest, identity)
        if dry_run:
            plan = preflight(config)
            write_json(directory / "preflight.json", plan)
            return {
                "run_dir": str(directory),
                "dry_run": True,
                "new_model_calls": 0,
                "ready": plan["ready"],
            }
        provider = GraftProvider(opts)
        provider.preflight()
        environment = {
            "provider": provider.metadata,
            "python": sys.version,
            "platform": platform.platform(),
        }
        envfile = directory / "environment.json"
        if envfile.exists() and json.loads(envfile.read_text()) != environment:
            raise CompatibilityError("Resume model digest/provider version/environment changed")
        write_json(envfile, environment)
        store = Store(directory, opts)
        statuses = []
        try:
            for instance in instances:
                path = directory / "instances" / safe_name(instance.instance_id)
                path.mkdir(parents=True, exist_ok=True)
                statusfile = path / "status.json"
                status = (
                    json.loads(statusfile.read_text())
                    if resume and statusfile.exists()
                    else {"instance_id": instance.instance_id, "arms": {}}
                )
                try:
                    if instance.provenance.get("skip_reason"):
                        status.update(status="skipped", error=instance.provenance["skip_reason"])
                        continue
                    i, entities, cs, metadata, depths, prepared = prepare_one(instance, opts)
                    if (path / "input.json").exists() and fingerprint(
                        json.loads((path / "input.json").read_text())
                    ) != fingerprint(prepared):
                        raise CompatibilityError("Prepared buggy sources or evidence changed")
                    graph = build_graph(i, entities)
                    write_json(path / "input.json", prepared)
                    write_json(path / "graph.json", graph.to_dict())
                    counter, embeddings = resources(config, i.instance_id)
                    if embeddings:
                        from rave_fl.graft.selection import query_text

                        for text in [n.text for n in graph.nodes] + [
                            query_text(graph, c.function_id) for c in cs
                        ]:
                            embeddings.lookup(text)
                    graft_config = GraftConfig(**config["graft"])
                    for fusion_name in opts["fusions"]:
                        for policy in ["prior"] + opts["policies"] + config["graft_policies"]:
                            key = safe_name(policy + "__" + fusion_name)
                            file = path / (key + ".json")
                            previous = status["arms"].get(key)
                            if previous and file.exists():
                                if file_hash(file) != previous["sha256"]:
                                    raise CompatibilityError("Saved arm checksum mismatch")
                                if previous["status"] == "completed":
                                    continue
                            budget = opts["policy_budgets"].get(policy, opts["budget"])
                            policy_opts = {**opts, "budget": budget}
                            fusion = Fusion(
                                fusion_name, opts["tie"], opts["rounding"], opts["rrf_k"]
                            )
                            backend, controller = None, None
                            row = {
                                "instance_id": i.instance_id,
                                "dataset": i.dataset,
                                "policy": policy,
                                "fusion": fusion_name,
                                "budget": budget,
                                "input_hash": fingerprint(prepared),
                                "graph_hash": graph.hash,
                                "synthetic": bool(provider.metadata.get("synthetic")),
                                "execution_kind": "synthetic_fixture"
                                if provider.metadata.get("synthetic")
                                else "fresh_or_exact_cache",
                                "cost_scope": "Verification provider input+output; excludes retrieval, embedding and benchmark tests",
                            }
                            try:
                                if policy == "prior":
                                    arm = {
                                        "ranked_functions": [
                                            c.function_id for c in sorted(cs, key=lambda c: c.rank)
                                        ],
                                        "actual_tokens": 0,
                                        "calls": [],
                                    }
                                elif policy in GRAFT_POLICIES:
                                    backend = Session(
                                        i,
                                        entities,
                                        provider,
                                        store,
                                        policy_opts,
                                        budget,
                                        graph,
                                        graft_config,
                                        counter,
                                    )
                                    controller = Router(cs, backend, fusion, policy, embeddings)
                                    arm = controller.execute()
                                else:
                                    backend = Backend(
                                        i, entities, provider, store, policy_opts, budget
                                    )
                                    controller = engine(
                                        cs, backend, policy, policy_opts, metadata, depths, fusion
                                    )
                                    arm = controller.execute()
                                row.update(
                                    arm=arm,
                                    status="completed"
                                    if policy == "prior" or complete(arm, policy, len(cs))
                                    else "incomplete",
                                    accounted_tokens=arm["actual_tokens"],
                                    action_accounting=backend.events if backend else [],
                                )
                            except Exception as exc:
                                row.update(
                                    status="incomplete"
                                    if isinstance(exc, (ExecutionLimit, UnknownUsage))
                                    else "failed",
                                    error=f"{type(exc).__name__}: {exc}",
                                    accounted_tokens=getattr(backend, "spent", 0),
                                    action_accounting=getattr(backend, "events", []),
                                    usage_unknown=store.snapshot()["usage_unknown"],
                                )
                                if isinstance(controller, Router):
                                    row["arm"] = controller.result("exception: " + str(exc))
                            write_json(file, row)
                            status["arms"][key] = {
                                "file": str(file.relative_to(directory)),
                                "sha256": file_hash(file),
                                "status": row["status"],
                            }
                            write_json(statusfile, status)
                    status["status"] = (
                        "completed"
                        if all(a["status"] == "completed" for a in status["arms"].values())
                        else "incomplete"
                    )
                except Exception as exc:
                    status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                finally:
                    write_json(statusfile, status)
                    statuses.append(status)
            # Dataset-specific gold extraction runs only after all decisions.
            if file_hash(source["dataset_file"]) != source["dataset_sha256"]:
                raise CompatibilityError("Dataset changed during collection")
            rows = []
            for instance, status in zip(instances, statuses):
                path = directory / "instances" / safe_name(instance.instance_id)
                try:
                    gold = load_gold(instance, source)
                    write_json(path / "evaluation_only.json", asdict(gold))
                except Exception as exc:
                    gold, gold_error = None, str(exc)
                entities = (
                    [
                        CodeEntity(**e)
                        for e in json.loads((path / "input.json").read_text())["entities"]
                    ]
                    if (path / "input.json").exists()
                    else []
                )
                for entry in status["arms"].values():
                    file = directory / entry["file"]
                    row = json.loads(file.read_text())
                    row["metrics"] = (
                        {
                            p: evaluate_profile(
                                row.get("arm", {"ranked_functions": []}),
                                entities,
                                gold,
                                instance.dataset,
                                p,
                            )
                            for p in PROFILES
                        }
                        if gold
                        else {}
                    )
                    row["gold_unavailable_reason"] = gold.unavailable_reason if gold else gold_error
                    write_json(file, row)
                    entry["sha256"] = file_hash(file)
                    rows.append(row)
                if not status["arms"]:
                    # Failed preparation is still an attempted defect for every
                    # configured comparison; do not shrink the denominator.
                    for policy in ["prior"] + opts["policies"] + config["graft_policies"]:
                        for fusion in opts["fusions"]:
                            metrics = (
                                {
                                    p: evaluate_profile(
                                        {"ranked_functions": []},
                                        entities,
                                        gold,
                                        instance.dataset,
                                        p,
                                    )
                                    for p in PROFILES
                                }
                                if gold
                                else {}
                            )
                            rows.append(
                                {
                                    "instance_id": instance.instance_id,
                                    "policy": policy,
                                    "fusion": fusion,
                                    "status": status["status"],
                                    "error": status.get("error"),
                                    "accounted_tokens": None,
                                    "metrics": metrics,
                                    "gold_unavailable_reason": gold.unavailable_reason
                                    if gold
                                    else gold_error,
                                }
                            )
                write_json(path / "status.json", status)
            write_json(directory / "results.json", rows)
            spending = store.snapshot()
            write_json(directory / "accounting.json", spending)
            write_json(directory / "execution_status.json", statuses)
            from rave_fl.graft.reporting import report

            report(directory)
        finally:
            write_json(directory / "accounting.json", store.snapshot())
            store.db.close()
        return {
            "run_dir": str(directory),
            "statuses": {s["instance_id"]: s["status"] for s in statuses},
            "spending": spending,
        }
