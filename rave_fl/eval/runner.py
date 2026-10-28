"""Prepare immutable inputs, orchestrate shared observations, and evaluate offline."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from rave_fl.code_sketch import prepare_candidates
from rave_fl.core import Candidate, Action
from rave_fl.pilot import write_json
from rave_fl.pilot_types import CandidateState, Observation
from rave_fl.eval.adapters import load_instances
from rave_fl.eval.backend import Backend, Provider, Store, ExecutionLimit, UnknownUsage
from rave_fl.eval.candidates import provide
from rave_fl.eval.fusion import Fusion
from rave_fl.eval.metrics import load_gold, evaluate, aggregate
from rave_fl.eval.policies import engine
from rave_fl.eval.representations import prepare_entities
from rave_fl.eval.schema import (
    OnlineInstance,
    CodeEntity,
    CompatibilityError,
    fingerprint,
    file_hash,
    safe_name,
)


def implementation_identity():
    root = Path(__file__).resolve().parents[2]
    files = sorted([*root.joinpath("rave_fl").rglob("*.py"), *root.joinpath("wp1").glob("*.py")])
    return {str(p.relative_to(root)): file_hash(p) for p in files}


@contextmanager
def run_lock(directory):
    path = Path(directory) / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CompatibilityError(
                "Another process owns this run; do not run/resume concurrently"
            )
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_checkout(instance):
    path = Path(instance.checkout)
    if not path.is_dir():
        raise CompatibilityError(
            "Buggy checkout missing: " + str(path) + "; prepare an immutable checkout first"
        )
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True
    )
    if proc.returncode:
        raise CompatibilityError("Checkout must be a Git repository with a recorded base commit")
    revision = proc.stdout.strip()
    if instance.revision and revision != instance.revision:
        raise CompatibilityError("Checkout HEAD differs from declared buggy/base revision")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty:
        raise CompatibilityError(
            "Tracked checkout files modified; preserve official buggy/base sources"
        )
    return replace(instance, revision=revision)


def prepare_one(instance, opts):
    instance = validate_checkout(instance)
    if opts.get("require_observed_failure") and instance.evidence.kind != "observed_failure":
        raise CompatibilityError(
            "Observed buggy failure evidence required; issue/trigger text is a separate evidence condition"
        )
    entities, provenance = provide(instance, opts)
    if not entities:
        raise CompatibilityError(
            "Candidate provider returned an empty pool; candidate-generation failure retained"
        )
    if opts["granularity"] == "file" and any(e.kind != "file" for e in entities):
        raise CompatibilityError(
            "File granularity needs file candidates, not silently converted methods"
        )
    if opts["representation"] == "file_compact" and opts["granularity"] != "file":
        raise CompatibilityError("file_compact requires --granularity file")
    if (
        opts["candidate_provider"] == "flexfl"
        and opts["representation"] == "l1"
        and opts["sketch_chars"] == 2200
    ):
        # Preserve exact legacy source, sketches and features for Time-25.
        metadata = {
            e.entity_id: {
                "methods": list(e.ranks),
                "ranks": e.ranks,
                "top_per_method": opts["top_per_method"],
            }
            for e in entities
        }
        cs = [
            Candidate(e.entity_id, e.entity_id, semantic_score=e.prior, coverage_score=0)
            for e in entities
        ]
        candidates = prepare_candidates(
            cs, {e.entity_id: e.source for e in entities}, metadata, instance.evidence.text
        )
        depths = {m: opts["top_per_method"] for m in ["SBIR", "Ochiai", "BoostN"]}
    else:
        candidates, metadata, depths = prepare_entities(
            instance, entities, opts["representation"], opts["sketch_chars"]
        )
    value = {
        "online": instance.to_dict(),
        "entities": [asdict(e) for e in entities],
        "candidates": [asdict(c) for c in candidates],
        "rank_metadata": metadata,
        "rank_depths": depths,
        "candidate_provider": provenance,
        "pool_hash": fingerprint([asdict(e) for e in entities]),
        "evidence_sha256": instance.evidence.sha256,
    }
    return instance, entities, candidates, metadata, depths, value


def materialize(instances, opts):
    """Explicitly requested local worktrees; no builds/tests/model calls."""
    if not opts.get("materialize_checkouts"):
        return
    cache = Path(opts["repo_cache"]).resolve()
    for i in instances:
        if Path(i.checkout).exists():
            continue
        mirror = cache / safe_name(i.repository)
        if not mirror.exists():
            if not opts.get("allow_downloads"):
                raise CompatibilityError(
                    "Local repository cache missing; clone explicitly or use --allow-downloads with prepare"
                )
            mirror.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--mirror",
                    "https://github.com/" + i.repository + ".git",
                    str(mirror),
                ],
                check=True,
            )
        if not i.revision:
            raise CompatibilityError("Worktree preparation requires a declared immutable revision")
        Path(i.checkout).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "--git-dir",
                str(mirror),
                "worktree",
                "add",
                "--detach",
                i.checkout,
                i.revision,
            ],
            check=True,
        )


class LegacyBackend:
    """Selected exact historical action replay; estimates never read outcomes."""

    def __init__(self, corpus, opts, store, budget):
        self.corpus = corpus
        self.backend = corpus.backend()
        self.opts = opts
        self.store = store
        self.budget = budget
        self.spent = 0
        self.events = []
        self.context_window = opts["context_window"] + (
            opts["output_tokens"] * opts["batch_size"] if opts["admission"] == "strict" else 0
        )
        self.reserved = None
        if opts["admission"] == "strict" and opts.get("reservation_spec"):
            from rave_fl.budget_reservation import ReservedBackend

            self.reserved = ReservedBackend(
                self.backend,
                budget,
                json.loads(Path(opts["reservation_spec"]).read_text()),
                corpus.provenance["model_digest"],
                corpus.settings,
            )

    def estimate(self, cs, action, level):
        if self.reserved:
            return self.reserved.estimate(cs, action, level)
        return (
            self.backend.estimate(cs, action, level)
            if self.opts["admission"] == "historical"
            else self.opts["context_window"] + self.opts["output_tokens"] * len(cs)
        )

    def execute(self, cs, action, level):
        reserved = self.estimate(cs, action, level)
        if reserved > self.budget - self.spent:
            raise ExecutionLimit("Historical selected action fails current admission")
        result = (self.reserved or self.backend).execute(cs, action, level)
        self.spent += result.total_tokens
        event = {
            "event": "selected_historical_replay",
            "candidate_ids": [c.function_id for c in cs],
            "predicted_tokens": self.backend.estimate(cs, action, level),
            "reserved_tokens": reserved,
            "actual_tokens": result.total_tokens,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "reuse": "historical_replay",
            "remaining": self.budget - self.spent,
        }
        self.events.append(event)
        with self.store.lock:
            self.store.event(event)
        return result


def historical_corpus(instance, entities, opts):
    from rave_fl.replay import ReplayCorpus

    case = {
        "bug": instance.instance_id,
        "flexfl_repo": instance.flexfl_repo,
        "checkout": instance.checkout,
    }
    if instance.evidence.kind != "trigger_description":
        case["evidence_file"] = instance.evidence.source
    corpus = ReplayCorpus(Path(opts["replay_run"]), case)
    expected = (
        opts["model"],
        opts["temperature"],
        opts["seed"],
        opts["context_window"],
        opts["output_tokens"],
    )
    if (
        corpus.settings != expected
        or opts["granularity"] != "method"
        or opts["representation"] != "l1"
    ):
        raise CompatibilityError(
            "Historical replay requires original model/settings/method L1 representations"
        )
    if [e.entity_id for e in entities] != [c.function_id for c in corpus.candidates] or any(
        e.source != c.source for e, c in zip(entities, corpus.candidates)
    ):
        raise CompatibilityError("Historical replay candidate pool/source mismatch")
    if opts["batch_size"] != len(corpus.groups[0]):
        raise CompatibilityError("Historical replay batch composition changed")
    return corpus


def frozen_arm(instance, policy, opts, prepared_hash, corpus):
    aliases = {
        "full": "full",
        "fixed_l1": "fixed_batch",
        "v2_utility": "repair_utility",
        "rank_gate": "rank_threshold",
        "v1_cost_aware": "adaptive",
    }
    if corpus:
        source = json.loads(Path(opts["replay_run"]).read_text())
        case = next(c for c in source["cases"] if c["bug"] == instance.instance_id)
        arm = case["arms"].get(aliases.get(policy, policy))
        if arm is None:
            raise CompatibilityError("Frozen observation state unavailable for " + policy)
        return deepcopy(arm)
    source = Path(opts.get("observations_dir") or "")
    if not source.is_dir():
        raise CompatibilityError("Frozen mode requires --observations-dir or --replay-run")
    previous = source / "instances" / safe_name(instance.instance_id)
    value = json.loads((previous / "input.json").read_text())
    if fingerprint(value) != prepared_hash:
        raise CompatibilityError(
            "Frozen state has different source/evidence/candidate/representation fingerprints"
        )
    candidates = sorted(previous.glob(safe_name(policy) + "__*.json"))
    if len(candidates) != 1:
        raise CompatibilityError(
            "Frozen source policy ambiguous; supply source run with one fusion"
        )
    prior_status = json.loads((previous / "status.json").read_text())
    entry = next(
        (
            v
            for v in prior_status["arms"].values()
            if v["file"] == str(candidates[0].relative_to(source))
        ),
        None,
    )
    if not entry or file_hash(candidates[0]) != entry["sha256"]:
        raise CompatibilityError("Frozen observation checksum mismatch")
    return json.loads(candidates[0].read_text())["arm"]


def collect_instance(instance, opts, directory, provider, store, resume=False):
    path = Path(directory) / "instances" / safe_name(instance.instance_id)
    path.mkdir(parents=True, exist_ok=True)
    status_path = path / "status.json"
    status = {"instance_id": instance.instance_id, "status": "preparing", "arms": {}}
    if resume and status_path.exists():
        status = json.loads(status_path.read_text())
    write_json(status_path, status)
    if instance.provenance.get("skip_reason"):
        status.update(
            status="skipped", error="Explicit manifest skip: " + instance.provenance["skip_reason"]
        )
        write_json(status_path, status)
        return status
    try:
        instance, entities, cs, metadata, depths, value = prepare_one(instance, opts)
        if (path / "input.json").exists() and fingerprint(
            json.loads((path / "input.json").read_text())
        ) != fingerprint(value):
            raise CompatibilityError("Resume inputs changed; use a new run")
        write_json(path / "input.json", value)
        corpus = historical_corpus(instance, entities, opts) if opts.get("replay_run") else None
        if corpus:
            replay_file = path / "replay_provenance.json"
            if replay_file.exists() and fingerprint(
                json.loads(replay_file.read_text())
            ) != fingerprint(corpus.provenance):
                raise CompatibilityError("Historical replay source/ledger provenance changed")
            write_json(replay_file, corpus.provenance)
        status.update(status="running", input_sha256=fingerprint(value))
        write_json(status_path, status)
        if opts["dry_run"]:
            plan = {
                "instance_id": instance.instance_id,
                "candidates": len(cs),
                "full_calls_upper": len(cs),
                "l1_batches": (len(cs) + opts["batch_size"] - 1) // opts["batch_size"],
                "new_provider_calls": 0,
            }
            status.update(status="prepared", resource_estimate=plan)
            write_json(status_path, status)
            return status
        for policy in opts["policies"]:
            for fusion_name in opts["fusions"]:
                name = safe_name(policy) + "__" + safe_name(fusion_name)
                file = path / (name + ".json")
                if resume and status["arms"].get(name, {}).get("status") == "completed":
                    if not file.exists() or file_hash(file) != status["arms"][name]["sha256"]:
                        raise CompatibilityError(
                            "Completed result checksum mismatch; do not silently replay altered output"
                        )
                    continue
                policy_opts = {**opts, "budget": opts["policy_budgets"].get(policy, opts["budget"])}
                fusion = Fusion(fusion_name, opts["tie"], opts["rounding"], opts["rrf_k"])
                start = time.monotonic()
                backend = None
                controller = None
                try:
                    if policy == "ranker_disagreement" and not any(
                        m.get("ranks") for m in metadata.values()
                    ):
                        raise CompatibilityError(
                            "Ranker disagreement requires imported/FlexFL SBIR/Ochiai/BoostN positions; this provider has none"
                        )
                    if instance.language not in {"java", "python"} and (
                        opts["granularity"] != "file"
                        or opts["representation"] != "full"
                        or policy != "full"
                    ):
                        raise CompatibilityError(
                            "This language supports file/full inspection only; register a structural parser for other policies"
                        )
                    if opts["evaluation_mode"] == "frozen":
                        arm = frozen_arm(instance, policy, opts, fingerprint(value), corpus)
                        states = {
                            i: CandidateState(
                                s["level"],
                                Observation(**s["observation"]) if s["observation"] else None,
                            )
                            for i, s in arm["states"].items()
                        }
                        scores = fusion.scores(cs, states)
                        arm["ranked_functions"] = [c.function_id for c in fusion.order(cs, scores)]
                        arm["ranking_scores"] = scores
                        arm["ranker"] = fusion.name
                        arm["budget_exceeded"] = arm["actual_tokens"] > policy_opts["budget"]
                    else:
                        backend = (
                            LegacyBackend(corpus, policy_opts, store, policy_opts["budget"])
                            if corpus
                            else Backend(
                                instance,
                                entities,
                                provider,
                                store,
                                policy_opts,
                                policy_opts["budget"],
                            )
                        )
                        controller = engine(
                            cs, backend, policy, policy_opts, metadata, depths, fusion
                        )
                        arm = controller.execute()
                    all_l1 = all(s["level"] >= 1 for s in arm["states"].values())
                    complete = (
                        arm["full_verified_count"] == len(cs)
                        if policy == "full"
                        else arm["stop_reason"] == "policy_complete"
                        if policy == "rank_gate"
                        else arm["screened_count"] == len(cs)
                    )
                    if policy == "full" and arm["failed_full_ids"]:
                        complete = False
                    if any(
                        not s["observation"] or not s["observation"]["valid"]
                        for s in arm["states"].values()
                        if s["level"] >= 0
                    ):
                        complete = False
                    result = {
                        "instance_id": instance.instance_id,
                        "dataset": instance.dataset,
                        "language": instance.language,
                        "repository": instance.repository,
                        "policy": policy,
                        "fusion": fusion_name,
                        "independent_bug_id": instance.provenance["independent_bug_id"],
                        "fusion_config": fusion.config(),
                        "evaluation_mode": opts["evaluation_mode"],
                        "admission": opts["admission"],
                        "status": "completed"
                        if complete and not arm["budget_exceeded"]
                        else "incomplete",
                        "arm": arm,
                        "accounted_tokens": arm["actual_tokens"],
                        "input_tokens": arm["actual_input_tokens"],
                        "output_tokens": arm["actual_output_tokens"],
                        "full_calls": arm["full_verified_count"],
                        "calls": arm["call_count"],
                        "wall_seconds": time.monotonic() - start,
                        "budget_exceeded": arm["budget_exceeded"],
                        "unused_budget": policy_opts["budget"] - arm["actual_tokens"],
                        "incomplete_l1_coverage": (not all_l1)
                        if policy
                        not in {"full", "rank_gate", "confidence", "v1_staged", "v1_cost_aware"}
                        else None,
                        "action_accounting": backend.events if backend else [],
                        "input_hash": fingerprint(value),
                        "pool_hash": value["pool_hash"],
                        "metrics": {},
                        "synthetic": bool(provider.metadata.get("synthetic")) and not corpus,
                        "cost_scope": "Accounted verification only; no new physical inference for frozen/historical replay",
                    }
                except Exception as exc:
                    result = {
                        "instance_id": instance.instance_id,
                        "dataset": instance.dataset,
                        "language": instance.language,
                        "repository": instance.repository,
                        "policy": policy,
                        "fusion": fusion_name,
                        "independent_bug_id": instance.provenance["independent_bug_id"],
                        "evaluation_mode": opts["evaluation_mode"],
                        "admission": opts["admission"],
                        "status": "incomplete"
                        if isinstance(exc, (ExecutionLimit, UnknownUsage))
                        else "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "accounted_tokens": getattr(
                            backend, "spent", getattr(controller, "spent", None)
                        ),
                        "input_hash": fingerprint(value),
                        "pool_hash": value["pool_hash"],
                        "metrics": {},
                        "usage_unknown": store.snapshot()["usage_unknown"],
                        "synthetic": provider.kind == "fake",
                        "action_accounting": getattr(backend, "events", []),
                        "full_calls": sum(
                            c.get("level") == 2 for c in getattr(controller, "calls", [])
                        ),
                        "calls": len(getattr(backend, "events", [])),
                        "wall_seconds": time.monotonic() - start,
                        "budget_exceeded": getattr(backend, "spent", 0) > policy_opts["budget"],
                    }
                if result.get("arm"):
                    ss = {c.function_id: CandidateState() for c in cs}
                    history = []
                    spent = 0
                    for step, call in enumerate([None] + result["arm"]["calls"]):
                        if call:
                            spent += call["input_tokens"] + call["output_tokens"]
                            for obs in call["observations"]:
                                ss[obs["function_id"]] = CandidateState(
                                    call["level"], Observation(**obs)
                                )
                        scores = fusion.scores(cs, ss)
                        for rank, candidate in enumerate(fusion.order(cs, scores), 1):
                            history.append(
                                {
                                    "step": step,
                                    "candidate_id": candidate.function_id,
                                    "rank": rank,
                                    "score": scores[candidate.function_id],
                                    "level": ss[candidate.function_id].level,
                                    "spent_tokens": spent,
                                }
                            )
                    result["score_trajectories"] = history
                write_json(file, result)
                status["arms"][name] = {
                    "status": result["status"],
                    "file": str(file.relative_to(directory)),
                    "sha256": file_hash(file),
                }
                write_json(status_path, status)
        status["status"] = (
            "completed"
            if all(v["status"] == "completed" for v in status["arms"].values())
            else "incomplete"
        )
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(status_path, status)
    return status


def evaluation(directory, instances, source, opts):
    if file_hash(source["dataset_file"]) != source["dataset_sha256"]:
        raise CompatibilityError(
            "Dataset changed during collection; preserve collected data and investigate before evaluating"
        )
    rows = []
    for instance in instances:
        path = Path(directory) / "instances" / safe_name(instance.instance_id)
        status = json.loads((path / "status.json").read_text())
        try:
            gold = load_gold(instance, source)
            gold_error = None
        except Exception as exc:
            gold = None
            gold_error = str(exc)
        prepared = (
            json.loads((path / "input.json").read_text())
            if (path / "input.json").exists()
            else None
        )
        entities = [CodeEntity(**e) for e in prepared["entities"]] if prepared else []
        if gold:
            write_json(path / "evaluation_only.json", asdict(gold))
        for policy in opts["policies"]:
            for fusion in opts["fusions"]:
                key = safe_name(policy) + "__" + safe_name(fusion)
                file = path / (key + ".json")
                if file.exists():
                    row = json.loads(file.read_text())
                else:
                    row = {
                        "instance_id": instance.instance_id,
                        "dataset": instance.dataset,
                        "language": instance.language,
                        "repository": instance.repository,
                        "policy": policy,
                        "fusion": fusion,
                        "admission": opts["admission"],
                        "evaluation_mode": opts["evaluation_mode"],
                        "status": status["status"],
                        "error": status.get("error", "No completed execution"),
                        "accounted_tokens": None,
                        "full_calls": 0,
                        "calls": 0,
                        "wall_seconds": None,
                        "metrics": {},
                    }
                if gold:
                    arm = row.get("arm", {"ranked_functions": []})
                    row["metrics"] = evaluate(arm, entities, gold, instance.dataset)
                    row["offline_gold_hash"] = fingerprint(asdict(gold))
                else:
                    row["metrics"] = {}
                    row["gold_error"] = gold_error or "Labels unavailable"
                if not row.get("arm"):
                    for granularity in ["file", "method"]:
                        metric = row["metrics"].get(granularity)
                        if metric:
                            for metric_key in list(metric):
                                if metric_key == "MRR" or metric_key.startswith(
                                    ("Top-", "Hit@", "Recall@")
                                ):
                                    metric[metric_key] = None
                row["metrics"]["gold_unavailable_reason"] = (
                    gold.unavailable_reason if gold else row["gold_error"]
                )
                if file.exists():
                    write_json(file, row)
                    status["arms"][key]["sha256"] = file_hash(file)
                rows.append(row)
        write_json(path / "status.json", status)
    write_json(Path(directory) / "results.json", rows)
    write_json(Path(directory) / "summary.json", aggregate(rows))
    write_json(
        Path(directory) / "audits/results.json",
        {
            "results_sha256": file_hash(Path(directory) / "results.json"),
            "dataset_sha256": source["dataset_sha256"],
            "rows": len(rows),
        },
    )
    return rows


def run(opts, resume_dir=None):
    if resume_dir:
        authorization = opts.get("allow_inference", False)
        directory = Path(resume_dir).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        opts = {**manifest["config"], "allow_inference": authorization}
        if (
            opts["backend"] != "fake"
            and not opts.get("replay_run")
            and opts["evaluation_mode"] != "frozen"
            and not opts["dry_run"]
            and not opts.get("allow_inference")
        ):
            raise CompatibilityError(
                "Resume needs explicit --allow-inference if new model calls may be required"
            )
        source = manifest["dataset_source"]
        instances = [OnlineInstance.from_dict(v) for v in manifest["instances"]]
        if fingerprint({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
            raise CompatibilityError("Run manifest checksum mismatch")
        if file_hash(source["dataset_file"]) != source["dataset_sha256"]:
            raise CompatibilityError("Resume dataset changed")
        if manifest.get("implementation") != implementation_identity():
            raise CompatibilityError(
                "Resume implementation changed; preserve this run and create a new run"
            )
    else:
        instances, source = load_instances(opts)
        directory = (
            Path(opts["output_dir"]).resolve()
            if opts.get("output_dir")
            else Path("results/rave_fl_eval") / opts["dataset"] / opts["run_id"]
        )
        if directory.exists():
            raise CompatibilityError(
                "Output exists; use resume or choose a new --output-dir. Historical runs are never overwritten."
            )
        directory.mkdir(parents=True)
        manifest = {
            "schema": "rave_eval_v1",
            "config": opts,
            "instances": [i.to_dict() for i in instances],
            "dataset_source": source,
            "selected_before_inference": True,
            "implementation": implementation_identity(),
        }
        manifest["sha256"] = fingerprint(manifest)
        write_json(directory / "manifest.json", manifest)
        write_json(
            directory / "reproduction_config.json",
            {
                **opts,
                "allow_inference": False,
                "allow_downloads": False,
                "materialize_checkouts": False,
            },
        )
        for instance in instances:
            write_json(
                directory / "instances" / safe_name(instance.instance_id) / "status.json",
                {"instance_id": instance.instance_id, "status": "pending", "arms": {}},
            )
    if (
        opts["backend"] != "fake"
        and not opts.get("replay_run")
        and opts["evaluation_mode"] != "frozen"
        and not opts["dry_run"]
        and not opts["allow_inference"]
    ):
        raise CompatibilityError(
            "Real inference disabled. Use --dry-run, --backend fake, or explicitly --allow-inference later."
        )
    with run_lock(directory):
        write_json(
            directory / "logs/execution.json",
            {
                "operation": "resume" if resume_dir else "run",
                "run_id": opts["run_id"],
                "evaluation_mode": opts["evaluation_mode"],
                "dry_run": opts["dry_run"],
                "selected_instances": [i.instance_id for i in instances],
                "real_inference_authorized": bool(opts["allow_inference"]),
                "benchmark_tests_executed": False,
            },
        )
        provider = Provider(opts)
        if opts.get("replay_run") or opts["evaluation_mode"] == "frozen" or opts["dry_run"]:
            provider.metadata = {
                "digest": opts.get("model_digest") or "recorded-input-validation",
                "provider_version": "no-inference",
                "synthetic": False,
            }
            if opts.get("replay_run") and not opts["dry_run"]:
                from rave_fl.replay import ReplayCorpus

                first = instances[0]
                case = {
                    "bug": first.instance_id,
                    "checkout": first.checkout,
                    "flexfl_repo": first.flexfl_repo,
                }
                if first.evidence.kind != "trigger_description":
                    case["evidence_file"] = first.evidence.source
                provenance = ReplayCorpus(Path(opts["replay_run"]), case).provenance
                provider.metadata = {
                    "name": opts["model"],
                    "digest": provenance["model_digest"],
                    "provider_version": "historical replay; original provider version not inferred",
                    "replay_provenance_sha256": fingerprint(provenance),
                    "synthetic": False,
                }
            if opts["evaluation_mode"] == "frozen" and not opts.get("replay_run"):
                source_dir = Path(opts.get("observations_dir") or "")
                previous_manifest = json.loads((source_dir / "manifest.json").read_text())
                fields = [
                    "model",
                    "temperature",
                    "seed",
                    "context_window",
                    "output_tokens",
                    "admission",
                    "budget",
                    "policy_budgets",
                    "max_full",
                ]
                if any(previous_manifest["config"][k] != opts[k] for k in fields):
                    raise CompatibilityError(
                        "Frozen observations must retain their original model/generation settings"
                    )
                provider.metadata = {
                    **json.loads((source_dir / "environment.json").read_text())["provider"],
                    "execution_kind": "frozen_reranking",
                }
        else:
            provider.preflight()
        envfile = directory / "environment.json"
        if envfile.exists():
            previous = json.loads(envfile.read_text())
            if previous["provider"] != provider.metadata:
                raise CompatibilityError("Resume provider/model digest/version changed")
        else:
            write_json(
                envfile,
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "provider": provider.metadata,
                    "device_requested": opts["device"],
                    "hardware_performance_measured": False,
                    "dependencies": {
                        name: importlib.metadata.version(name)
                        for name in ["numpy", "matplotlib", "PyYAML"]
                    },
                    "source_revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=Path(__file__).resolve().parents[2],
                        text=True,
                    ).strip(),
                },
            )
        if resume_dir:
            for item in instances:
                existing = directory / "instances" / safe_name(item.instance_id)
                if (existing / "input.json").exists():
                    current = prepare_one(item, opts)[-1]
                    if fingerprint(current) != fingerprint(
                        json.loads((existing / "input.json").read_text())
                    ):
                        raise CompatibilityError(
                            "Resume source/evidence/candidate inputs changed; create a new run"
                        )
                if (existing / "status.json").exists():
                    saved = json.loads((existing / "status.json").read_text())
                    for arm in saved.get("arms", {}).values():
                        if (
                            arm["status"] == "completed"
                            and file_hash(directory / arm["file"]) != arm["sha256"]
                        ):
                            raise CompatibilityError("Completed result checksum mismatch")
        store = Store(directory, opts)
        with ThreadPoolExecutor(max_workers=opts["workers"]) as pool:
            statuses = list(
                pool.map(
                    lambda i: collect_instance(
                        i, opts, directory, provider, store, bool(resume_dir)
                    ),
                    instances,
                )
            )
        write_json(
            directory / "execution_status.json",
            {"instances": statuses, "spending": store.snapshot(), "dry_run": opts["dry_run"]},
        )
        if not opts["dry_run"]:
            evaluation(directory, instances, source, opts)
            from rave_fl.eval.reporting import report

            report(directory)
        write_json(
            directory / "audits/accounting.json",
            {
                "spending": store.snapshot(),
                "status": "usage_unknown"
                if store.snapshot()["usage_unknown"]
                else "counter_journal_available",
                "real_llm_calls": 0
                if opts["backend"] == "fake"
                or opts.get("replay_run")
                or opts["evaluation_mode"] == "frozen"
                else store.snapshot()["new_calls"],
            },
        )
    return {
        "run_dir": str(directory),
        "instances": len(instances),
        "statuses": {i["instance_id"]: i["status"] for i in statuses},
        "spending": store.snapshot(),
    }
