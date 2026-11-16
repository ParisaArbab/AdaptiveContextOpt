"""Isolate Phase-3 preflight crashes by bug and stage, with no model inference.

Each frozen bug is prepared in its own Python process. A native fault leaves a
faulthandler traceback and the last Java filename in that bug's log. The tool
does not alter official checkouts, execute tests, or call Qwen.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

BUGS = ("Lang-1", "Math-1", "Time-1")


def diagnose_one(config_path: Path, bug: str) -> int:
    # Isolate heavyweight native AST parsing to this short-lived process.
    from rave_fl.eval.adapters import load_instances
    from rave_fl.eval.runner import prepare_one
    from rave_fl.graft.graph import build_graph
    from rave_fl.graft.runner import resources, validate_config

    config = validate_config(json.loads(config_path.read_text()))
    b = config["benchmark"]
    if b["candidate_provider"] != "bm25" or set(b["bugs"]) != set(BUGS):
        raise ValueError("Only frozen three-bug BM25 Phase-3 diagnostic is supported")

    print(f"[diagnostic] bug={bug}: loading real observed-test manifest", flush=True)
    instances, _ = load_instances(b)
    if {inst.instance_id for inst in instances} != set(BUGS):
        raise ValueError("Frozen 3-bug cohort incomplete")
    instance = next(i for i in instances if i.instance_id == bug)
    if instance.evidence.kind != "observed_failure":
        raise ValueError("Observed Defects4J failure evidence required")

    print(f"[diagnostic] bug={bug}: candidate indexing and preparation", flush=True)
    prepared, entities, candidates, _, _, _ = prepare_one(instance, b)
    print(
        f"[diagnostic] bug={bug}: prepared {len(entities)} candidates, graph building",
        flush=True,
    )
    graph = build_graph(prepared, entities)
    print(f"[diagnostic] bug={bug}: graph nodes={len(graph.nodes)}, resources", flush=True)
    counter, embeddings = resources(config, prepared.instance_id)
    if embeddings:
        from rave_fl.graft.selection import query_text
        for query in [n.text for n in graph.nodes] + [
            query_text(graph, c.function_id) for c in candidates
        ]:
            embeddings.lookup(query)
    print(f"[diagnostic] bug={bug}: OK, counter={counter.name}", flush=True)
    return 0


def run_isolated(config_path: Path, output: Path, timeout: int) -> int:
    if output.exists():
        raise FileExistsError(f"Diagnostic directory exists, refusing overwrite: {output}")
    output.mkdir(parents=True)
    config_path = config_path.resolve()
    failures = []
    for bug in BUGS:
        log = output / f"{bug}.preflight.log"
        print(f"Diagnosing {bug}, writing {log}", flush=True)
        env = dict(os.environ)
        env["GRAFT_TRACE_JAVA_INDEX"] = "1"
        args = [
            sys.executable, "-X", "faulthandler",
            "-m", "scripts.diagnose_graft_phase3_preflight",
            "--config", str(config_path), "--bug", bug,
        ]
        try:
            with log.open("w") as handle:
                result = subprocess.run(
                    args,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                    timeout=timeout,
                )
            status = result.returncode
            status_text = "passed" if status == 0 else (
                "native signal " + str(-status) if status < 0
                else "exit code " + str(status)
            )
        except subprocess.TimeoutExpired:
            status = -999
            status_text = f"timed out after {timeout}s"
        print(f"{bug}: {status_text}", flush=True)
        if status != 0:
            failures.append(bug)
            # The last printed source file and faulthandler stack locate
            # the failure. Keep the entire log unmodified on disk.
            lines = log.read_text(errors="replace").splitlines()
            print(f"---- {bug}: last 50 diagnostic lines ----", flush=True)
            print("\n".join(lines[-50:]), flush=True)
            print(f"---- complete log: {log} ----", flush=True)
    print(
        "No inference, provider requests or test executions performed. "
        + ("Failures: " + ", ".join(failures) if failures else "All bugs passed."),
        flush=True,
    )
    return 3 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bug", choices=BUGS, help="Internal isolated diagnostic")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error("Missing prepared Phase-3 config")
    if args.bug:
        return diagnose_one(args.config, args.bug)
    if args.output is None:
        parser.error("Supply --output diagnostic folder")
    if not 30 <= args.timeout <= 3600:
        parser.error("Timeout must be between 30 and 3600 seconds")
    return run_isolated(args.config, args.output, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
