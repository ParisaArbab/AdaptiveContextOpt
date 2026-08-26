"""
graphify_structure.py — WP1, pipeline step 0

Runs BEFORE the control/rtk/lean-ctx comparison. For each instance's repo
checkout, runs the real `graphify` CLI (Graphify-Labs/graphify, pip package
`graphifyy`) in --code-only mode: fully local tree-sitter AST parsing, no LLM
call, nothing leaves the machine. This gives every condition (raw, rtk,
lean-ctx) the SAME structural map, so Graphify is not an unfair advantage
specific to one compression condition — it isolates the compression variable.

Verified real output (tested against psf/requests as a smoke test):
  `graphify extract . --code-only --no-viz` writes graphify-out/graph.json,
  a networkx node-link JSON: {"nodes": [...], "links": [...], ...}. Each node
  has: id, label, source_file, source_location ("L84"), community,
  _callable, _callable_class.

We reduce that full graph down to a compact per-instance structure map:
  { "path/to/file.py::FuncOrClass": {"file":..., "line": 84, "community": 0} }
This is what gets attached to the agent's context in every condition, and
also what agent_localizer.py can point Graphify's own `graphify explain` /
`graphify path` commands at if deeper traversal is needed mid-run.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict


def build_structure_map(repo_path: Path, out_dir: Path | None = None) -> Dict[str, dict]:
    """Run graphify extract --code-only on repo_path, return a compact map.

    Raises RuntimeError with the real graphify stderr on failure — no silent
    fallback, since a structure map built from a bad extraction would quietly
    bias every downstream condition the same way, which is worse than
    failing loudly.
    """
    repo_path = Path(repo_path)
    out_dir = out_dir or (repo_path / "graphify-out")
    cmd = ["graphify", "extract", str(repo_path), "--code-only", "--no-viz"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"graphify extract failed for {repo_path}:\n{result.stderr}"
        )

    graph_json = repo_path / "graphify-out" / "graph.json"
    if not graph_json.exists():
        raise RuntimeError(f"graphify reported success but {graph_json} is missing")

    graph = json.loads(graph_json.read_text())
    structure: Dict[str, dict] = {}
    for node in graph.get("nodes", []):
        source_file = node.get("source_file")
        label = node.get("label")
        if not source_file or not label:
            continue
        loc = node.get("source_location", "")
        line = int(loc.lstrip("L")) if loc.startswith("L") and loc[1:].isdigit() else None
        key = f"{source_file}::{label}"
        structure[key] = {
            "id": node.get("id"),
            "file": source_file,
            "line": line,
            "community": node.get("community"),
            "callable": bool(node.get("_callable")),
        }

    return structure


def build_call_graph(repo_path: Path) -> Dict[str, Dict[str, list]]:
    """Real GraphLocator-style substrate: an adjacency map built from
    Graphify's actual 'calls' edges (graph.json 'links'), not a proxy like
    community clustering.

    GraphLocator's real mechanism (verified against the paper): a causal
    issue graph (CIG) whose vertices are code entities and whose edges are
    causal/call dependencies; workflow = (1) locate symptom vertices, then
    (2) dynamically expand the CIG by iteratively reasoning over NEIGHBORING
    VERTICES ON THE REPOSITORY GRAPH. That repository graph, for us, is
    exactly Graphify's call graph — so this function returns each node's
    real callers and callees, which agent_localizer.py walks outward from
    the symptom vertices (stack-trace-evidenced methods).

    Returns: {node_id: {"callers": [node_id, ...], "callees": [node_id, ...]}}
    """
    repo_path = Path(repo_path)
    graph_json = repo_path / "graphify-out" / "graph.json"
    if not graph_json.exists():
        raise RuntimeError(f"{graph_json} missing — run build_structure_map first")

    graph = json.loads(graph_json.read_text())
    adjacency: Dict[str, Dict[str, list]] = {}

    def ensure(node_id: str):
        if node_id not in adjacency:
            adjacency[node_id] = {"callers": [], "callees": []}

    for link in graph.get("links", []):
        if link.get("relation") != "calls":
            continue
        src, dst = link.get("source"), link.get("target")
        if not src or not dst:
            continue
        ensure(src)
        ensure(dst)
        adjacency[src]["callees"].append(dst)
        adjacency[dst]["callers"].append(src)

    return adjacency


def id_to_key_map(structure: Dict[str, dict]) -> Dict[str, str]:
    """structure_map is keyed by 'file::label'; the call graph is keyed by
    Graphify's internal node id. This inverts structure_map for lookups."""
    return {meta["id"]: key for key, meta in structure.items() if meta.get("id")}


def save_structure_map(repo_path: Path, dest: Path) -> Dict[str, dict]:
    structure = build_structure_map(repo_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(structure, indent=2))
    return structure


def format_for_agent(structure: Dict[str, dict], max_entries: int = 400) -> str:
    """Compact text block handed to the localization agent alongside the
    (raw / rtk / lean-ctx) tool output. Same text, every condition."""
    lines = ["# Repository structure map (graphify, tree-sitter AST, local-only)"]
    for i, (key, meta) in enumerate(structure.items()):
        if i >= max_entries:
            lines.append(f"... ({len(structure) - max_entries} more entries truncated)")
            break
        loc = f"{meta['file']}:{meta['line']}" if meta["line"] else meta["file"]
        lines.append(f"- {key}  [{loc}]  community={meta['community']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("repo_path", type=str)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    repo = Path(args.repo_path)
    out = Path(args.out) if args.out else repo / "graphify-out" / "structure_map.json"
    structure = save_structure_map(repo, out)
    print(f"wrote {len(structure)} structural entries -> {out}")
