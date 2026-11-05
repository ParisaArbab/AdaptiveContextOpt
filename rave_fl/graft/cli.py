"""Reproducible GRAFT entry point. Real inference is explicitly opt-in."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from rave_fl.pilot import write_json
from rave_fl.graft.graph import Graph, build_graph
from rave_fl.graft.controller import GraftConfig, select_evidence
from rave_fl.graft.selection import TokenizerCounter
from rave_fl.graft.topology import EmbeddingBundle
from rave_fl.graft.runner import validate_config, preflight, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "fixture", help="Create committed synthetic Java/Python inputs; no model calls"
    )
    p.add_argument("--output", required=True)
    for command in ["preflight", "run"]:
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "run":
            p.add_argument("--output", required=True)
            p.add_argument("--allow-inference", action="store_true")
            p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("resume")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--allow-inference", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser(
        "diagnostics",
        help="Regenerate archive structural diagnostics, tables and figures; no inference",
    )
    p.add_argument("--archive", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("graph")
    p.add_argument(
        "--input", required=True, help="Prepared input.json containing typed online and entities"
    )
    p.add_argument("--output", required=True)
    p = sub.add_parser("select")
    p.add_argument("--graph", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--input-budget", type=int, default=4096)
    p.add_argument("--tokenizer-dir")
    p.add_argument("--selector", choices=["bm25", "topo"], default="bm25")
    p.add_argument("--embeddings")
    p.add_argument("--paths", action="store_true")
    p.add_argument("--no-repair", action="store_true")
    p.add_argument("--typed-topology", action="store_true")
    p.add_argument("--no-propagation", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "fixture":
            from rave_fl.eval.fixtures import create

            manifest = create(args.output)
            config = {
                "benchmark": {
                    "dataset": "custom",
                    "dataset_file": str(manifest),
                    "backend": "fake",
                    "model": "synthetic-fixture",
                    "context_window": 8192,
                    "budget": 24000,
                    "batch_size": 2,
                    "max_full": 1,
                    "physical_budget": 100000,
                    "per_bug_physical_budget": 50000,
                    "max_new_calls": 40,
                },
                "graft": asdict(GraftConfig()),
                "graft_policies": ["l1_graph", "graft_adaptive"],
            }
            target = Path(args.output) / "graft.json"
            write_json(target, config)
            result = {"config": str(target), "synthetic": True, "new_model_calls": 0}
        elif args.command == "graph":
            from rave_fl.eval.schema import OnlineInstance, CodeEntity

            value = json.loads(Path(args.input).read_text())
            graph = build_graph(
                OnlineInstance.from_dict(value["online"]),
                [CodeEntity(**e) for e in value["entities"]],
            )
            write_json(args.output, graph.to_dict())
            result = {"graph_hash": graph.hash, "nodes": len(graph.nodes), "new_model_calls": 0}
        elif args.command == "select":
            graph = Graph.from_dict(json.loads(Path(args.graph).read_text()))
            config = GraftConfig(
                selector=args.selector,
                input_budget=args.input_budget,
                paths=args.paths,
                repair=not args.no_repair,
                typed_topology=args.typed_topology,
                propagation=not args.no_propagation,
            )
            counter = TokenizerCounter(args.tokenizer_dir) if args.tokenizer_dir else None
            bundle = EmbeddingBundle.load(args.embeddings) if args.embeddings else None
            selected = select_evidence(graph, args.target, config, counter, bundle)
            write_json(args.output, selected.to_dict())
            result = {
                "status": selected.status,
                "input_units": selected.input_units,
                "counter": selected.counter,
                "selection_hash": selected.hash,
                "reason": selected.reason,
                "new_model_calls": 0,
            }
        elif args.command == "diagnostics":
            from rave_fl.graft.diagnostics import generate

            result = generate(args.archive, args.output)
        elif args.command == "report":
            from rave_fl.graft.reporting import report

            result = report(args.run_dir)
        elif args.command == "resume":
            saved = json.loads((Path(args.run_dir) / "manifest.json").read_text())
            result = run(
                saved["config"],
                args.run_dir,
                resume=True,
                allow_inference=args.allow_inference,
                dry_run=saved["dry_run"],
            )
        else:
            config = validate_config(json.loads(Path(args.config).read_text()))
            result = (
                preflight(config)
                if args.command == "preflight"
                else run(
                    config, args.output, allow_inference=args.allow_inference, dry_run=args.dry_run
                )
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        statuses = result.get("statuses", {})
        return (
            2
            if any(v == "failed" for v in statuses.values())
            else 3
            if any(v != "completed" for v in statuses.values())
            or result.get("ready") is False
            or result.get("status") == "infeasible"
            else 0
        )
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
        print("GRAFT compatibility/execution error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
