"""Universal benchmark CLI. Default operations never invoke real inference."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from rave_fl.eval.adapters import DATASETS, capabilities, load_instances
from rave_fl.eval.candidates import PROVIDERS
from rave_fl.eval.fusion import FUSIONS
from rave_fl.eval.policies import POLICIES, DEFAULT_POLICIES
from rave_fl.eval.schema import CompatibilityError
from rave_fl.pilot import write_json

DEFAULTS = dict(
    dataset="custom",
    dataset_version=None,
    dataset_file=None,
    bugs=[],
    projects=[],
    limit=None,
    sample="first",
    seed=0,
    language=None,
    source_language=None,
    source_root=None,
    repository=None,
    checkouts_dir=None,
    repo_cache="data/rave_fl_repo_cache",
    candidate_provider="bm25",
    candidate_file=None,
    candidate_factory=None,
    candidate_count=24,
    top_per_method=10,
    granularity="method",
    representation="l1",
    sketch_chars=2200,
    max_source_files=3000,
    max_source_bytes=30000000,
    backend="ollama",
    backend_factory=None,
    endpoint=os.environ.get("RAVE_MODEL_ENDPOINT", "http://127.0.0.1:11434"),
    model="llama3.1:8b",
    model_digest=None,
    device="auto",
    context_window=16384,
    temperature=0.0,
    output_tokens=160,
    timeout=180.0,
    physical_budget=80000,
    per_bug_physical_budget=60000,
    budget=16000,
    policy_budgets={},
    max_new_calls=100,
    policies=list(DEFAULT_POLICIES),
    fusions=[FUSIONS[0]],
    max_full=1,
    batch_size=4,
    max_steps=200,
    admission="historical",
    reservation_spec=None,
    tie="original",
    rounding=None,
    rrf_k=60,
    output_dir=None,
    run_id=None,
    workers=1,
    cache_dir=os.environ.get("RAVE_CACHE_DIR"),
    cache_mode="use",
    dry_run=False,
    allow_inference=False,
    evaluation_mode="policy",
    replay_run=None,
    observations_dir=None,
    check_provider=False,
    materialize_checkouts=False,
    allow_downloads=False,
    split_manifest=None,
    split=None,
    require_observed_failure=False,
)


def options(parser):
    for key in [
        "dataset_file",
        "dataset_version",
        "checkouts_dir",
        "repo_cache",
        "source_language",
        "source_root",
        "language",
        "repository",
        "candidate_file",
        "candidate_factory",
        "backend_factory",
        "endpoint",
        "model",
        "model_digest",
        "reservation_spec",
        "output_dir",
        "run_id",
        "cache_dir",
        "replay_run",
        "observations_dir",
        "split_manifest",
        "split",
    ]:
        parser.add_argument("--" + key.replace("_", "-"))
    for key in [
        "limit",
        "seed",
        "candidate_count",
        "top_per_method",
        "sketch_chars",
        "max_source_files",
        "max_source_bytes",
        "context_window",
        "output_tokens",
        "physical_budget",
        "per_bug_physical_budget",
        "budget",
        "max_new_calls",
        "max_full",
        "batch_size",
        "max_steps",
        "workers",
        "rounding",
        "rrf_k",
    ]:
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--config", help="JSON or YAML configuration; CLI flags override explicit fields"
    )
    parser.add_argument("--dataset", choices=DATASETS)
    for key, choices in [
        ("candidate_provider", PROVIDERS),
        ("backend", ["ollama", "fake", "plugin"]),
        ("device", ["auto", "cpu", "apple", "cuda", "a100"]),
        ("granularity", ["file", "method"]),
        ("representation", ["full", "l1", "file_compact"]),
        ("admission", ["historical", "strict"]),
        ("tie", ["original", "identifier", "hash"]),
        ("cache_mode", ["use", "readonly", "off"]),
        ("sample", ["first", "random"]),
        ("evaluation_mode", ["policy", "frozen"]),
    ]:
        parser.add_argument("--" + key.replace("_", "-"), choices=choices)
    for key in ["bugs", "projects", "policies", "fusions"]:
        parser.add_argument("--" + key, help="Comma-separated values")
    parser.add_argument("--policy-budgets", help="Comma-separated name=tokens overrides")
    for key in [
        "dry_run",
        "allow_inference",
        "check_provider",
        "materialize_checkouts",
        "allow_downloads",
        "require_observed_failure",
    ]:
        parser.add_argument("--" + key.replace("_", "-"), action="store_true", default=None)


def configuration(namespace):
    args = vars(namespace)
    opts = dict(DEFAULTS)
    if args.get("config"):
        p = Path(args["config"])
        if p.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(p.read_text())
        else:
            data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise CompatibilityError("Configuration must be a mapping")
        if set(data) - set(DEFAULTS):
            raise CompatibilityError(
                "Unknown configuration fields: " + str(set(data) - set(DEFAULTS))
            )
        opts.update(data)
    opts.update({k: v for k, v in args.items() if k in DEFAULTS and v is not None})
    for k in ["bugs", "projects", "policies", "fusions"]:
        if isinstance(opts[k], str):
            opts[k] = [s.strip() for s in opts[k].split(",") if s.strip()]
        if not isinstance(opts[k], list) or any(not isinstance(v, str) for v in opts[k]):
            raise CompatibilityError(k + " must be a list or comma-separated string")
    if isinstance(opts["policy_budgets"], str):
        try:
            opts["policy_budgets"] = {
                p.split("=", 1)[0]: int(p.split("=", 1)[1])
                for p in opts["policy_budgets"].split(",")
                if p
            }
        except (ValueError, IndexError):
            raise CompatibilityError("policy-budgets requires policy=tokens pairs")
    if not isinstance(opts["policy_budgets"], dict):
        raise CompatibilityError("policy_budgets must be a mapping")
    for key in [
        "dry_run",
        "allow_inference",
        "check_provider",
        "materialize_checkouts",
        "allow_downloads",
        "require_observed_failure",
    ]:
        if type(opts[key]) is not bool:
            raise CompatibilityError(key + " must be a boolean, not quoted text")
    for k in [
        "candidate_count",
        "top_per_method",
        "sketch_chars",
        "max_source_files",
        "max_source_bytes",
        "context_window",
        "output_tokens",
        "physical_budget",
        "per_bug_physical_budget",
        "max_new_calls",
        "batch_size",
        "max_steps",
        "workers",
        "rrf_k",
    ]:
        if type(opts[k]) is not int or opts[k] < 1:
            raise CompatibilityError(k + " must be a positive integer")
    if opts["limit"] is not None and (type(opts["limit"]) is not int or opts["limit"] < 1):
        raise CompatibilityError("limit must be positive")
    if any(type(opts[k]) is not int or opts[k] < 0 for k in ["budget", "max_full"]):
        raise CompatibilityError("Budget/call limit must be nonnegative integers")
    if type(opts["seed"]) is not int:
        raise CompatibilityError("seed must be an integer")
    if set(opts["policies"]) - set(POLICIES) or not opts["policies"]:
        raise CompatibilityError("Unknown/empty policy set")
    if set(opts["fusions"]) - set(FUSIONS) or not opts["fusions"]:
        raise CompatibilityError("Unknown/empty fusion set")
    if len(set(opts["policies"])) != len(opts["policies"]) or len(set(opts["fusions"])) != len(
        opts["fusions"]
    ):
        raise CompatibilityError("Duplicate policies/fusions are not independent runs")
    if set(opts["policy_budgets"]) - set(opts["policies"]) or any(
        type(v) is not int or v < 0 for v in opts["policy_budgets"].values()
    ):
        raise CompatibilityError(
            "Per-policy budget overrides must name selected policies and use nonnegative integers"
        )
    if opts["run_id"] is None:
        opts["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if (
        not isinstance(opts["run_id"], str)
        or not opts["run_id"]
        or opts["run_id"] in {".", ".."}
        or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for ch in opts["run_id"]
        )
    ):
        raise CompatibilityError("run-id must be a safe filename")
    if (
        opts["dataset"] not in DATASETS
        or opts["backend"] not in {"ollama", "fake", "plugin"}
        or opts["candidate_provider"] not in PROVIDERS
        or opts["admission"] not in {"historical", "strict"}
    ):
        raise CompatibilityError("Unknown dataset/backend/provider/admission in configuration")
    choices = {
        "device": {"auto", "cpu", "apple", "cuda", "a100"},
        "granularity": {"file", "method"},
        "representation": {"full", "l1", "file_compact"},
        "tie": {"original", "identifier", "hash"},
        "cache_mode": {"use", "readonly", "off"},
        "sample": {"first", "random"},
        "evaluation_mode": {"policy", "frozen"},
    }
    for key, allowed in choices.items():
        if opts[key] not in allowed:
            raise CompatibilityError("Unsupported " + key + " in configuration")
    if opts["rounding"] is not None and (
        type(opts["rounding"]) is not int or not 0 <= opts["rounding"] <= 12
    ):
        raise CompatibilityError("rounding must be an integer from 0 to 12")
    if opts["temperature"] < 0 or opts["timeout"] <= 0:
        raise CompatibilityError("Invalid temperature or timeout")
    return opts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-datasets", help="Print adapter capabilities; no network/inference")
    for command in ["preflight", "prepare", "run"]:
        p = sub.add_parser(
            command,
            help={
                "run": "Execute explicitly authorized live, synthetic, historical replay or dry-run evaluation",
                "prepare": "Validate local sources or explicitly materialize local worktrees; never tests/inference",
                "preflight": "Read-only resource/capability estimates; optional GET-only provider checks",
            }[command],
        )
        options(p)
    p = sub.add_parser(
        "resume", help="Resume immutable run; explicit authorization needed for new real calls"
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--allow-inference", action="store_true")
    p = sub.add_parser("report", help="Regenerate saved tables/figures; never inference")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser(
        "compare", help="Compare saved compatible runs or label descriptive settings"
    )
    p.add_argument("--runs", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--descriptive", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "list-datasets":
            result = capabilities()
        elif args.command == "report":
            from rave_fl.eval.reporting import report

            result = report(args.run_dir)
        elif args.command == "compare":
            from rave_fl.eval.reporting import compare

            result = compare(
                [Path(p) for p in args.runs.split(",")], args.output_dir, args.descriptive
            )
        elif args.command == "resume":
            from rave_fl.eval.runner import run

            result = run({"allow_inference": args.allow_inference}, args.run_dir)
        else:
            opts = configuration(args)
            if args.command == "preflight":
                from rave_fl.eval.preflight import inspect

                result = inspect(opts)
            else:
                from rave_fl.eval.runner import run, materialize

                if args.command == "prepare":
                    instances, _ = load_instances(opts)
                    materialize(instances, opts)
                    opts["dry_run"] = True
                result = run(opts)
        print(json.dumps(result, indent=2, allow_nan=False))
        statuses = result.get("statuses", {})
        return (
            2
            if any(v == "failed" for v in statuses.values())
            else 3
            if any(v in {"incomplete", "skipped"} for v in statuses.values())
            else 0
        )
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"RAVE-FL compatibility/execution error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
