"""RAVE-FL on one *real* SWE-bench Python checkout, not the Defects4J Java pilot.

Uses the existing SWE-bench workspace produced by this repository:
  data/swebench_workspaces/<instance>/repo
  data/swebench_workspaces/<instance>/metadata.json
  data/swebench_workspaces/<instance>/outputs/raw_test_output.txt

Run a gold-free dry-run before connecting real Ollama. Candidate retrieval
is explicitly a lexical production-code baseline, not SieveFL or SBIR.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from rave_fl.adaptive import AdaptivePilot
from rave_fl.code_sketch import prepare_candidates, words
from rave_fl.core import Candidate, Action
from rave_fl.pilot_backend import OllamaPilotBackend
from rave_fl.pilot_types import PolicyConfig
from rave_fl.rank_repair import REPAIR_MODES, RepairConfig, RankRepairPilot


def python_sketch(function_id: str, source: str, max_chars: int = 2200):
    """Real Python AST sketch with L0 signature, L1 control/data, L2 full code."""
    if not source.strip() or max_chars < 100:
        raise ValueError("Python sketch requires real nonempty source")
    try:
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        if not functions and not classes:
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        fn = functions[0] if functions else None
        if fn is None and classes:
            cls = classes[0]
            bases = ", ".join(ast.unparse(b) for b in cls.bases)
            l0 = f"CLASS: {function_id}\nDECLARATION: class {cls.name}({bases})"
            attributes = []
            calls = 0
            for n in cls.body:
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    attributes.append(ast.unparse(n))
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    attributes.append("METHOD SIGNATURE: " + n.name + "(" + ast.unparse(n.args) + ")")
                    calls += 1
            l1 = l0 + "\nCLASS STRUCTURE (lossy):\n" + "\n".join(attributes)
            truncated = len(l1) > max_chars
            if truncated:
                l1 = l1[:max_chars] + "\n[SKETCH TRUNCATED: verify full class]"
            return (l0, l1, source), {
                "parser": "python-ast-class", "parse_error": False,
                "truncated": truncated, "complexity": 1 + calls,
                "call_count": calls, "literal_count": 0,
            }
        if fn is None:
            raise ValueError("No Python callable or class in the provided source")
        prefix = "async def " if isinstance(fn, ast.AsyncFunctionDef) else "def "
        signature = prefix + fn.name + "(" + ast.unparse(fn.args) + ")"
        if fn.returns:
            signature += " -> " + ast.unparse(fn.returns)
        l0 = f"METHOD: {function_id}\nSIGNATURE: {signature}"
        selected = []
        complexity = 1
        calls = 0
        literals = 0
        for n in ast.walk(fn):
            if n is not fn and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                condition = getattr(n, "test", getattr(n, "iter", None))
                selected.append(type(n).__name__ + ": " + ast.unparse(condition))
                complexity += 1
            elif isinstance(n, ast.ExceptHandler):
                selected.append("except: " + (ast.unparse(n.type) if n.type else "all"))
                complexity += 1
            elif isinstance(n, (ast.Return, ast.Raise, ast.Assign, ast.AugAssign)):
                selected.append(ast.unparse(n))
            elif isinstance(n, ast.Call):
                calls += 1
                selected.append("CALL: " + ast.unparse(n.func))
            elif isinstance(n, ast.Constant):
                literals += 1
        unique = list(dict.fromkeys(selected))
        l1 = l0 + "\nSTRUCTURE (lossy, Python AST):\n" + "\n".join(unique)
        truncated = len(l1) > max_chars
        if truncated:
            l1 = l1[:max_chars] + "\n[SKETCH TRUNCATED: verify full source]"
        return (l0, l1, source), {
            "parser": "python-ast", "parse_error": False, "truncated": truncated,
            "complexity": complexity, "call_count": calls, "literal_count": literals,
        }
    except (SyntaxError, ValueError, RecursionError, TypeError) as exc:
        # Do not trust a failed sketch: the existing decision engine promotes it
        # directly to Full after discovery. Exact original source remains L2.
        return (f"METHOD: {function_id}", f"METHOD: {function_id}\n[PARSE ERROR: needs full source]", source), {
            "parser": "python-ast", "parse_error": True, "truncated": False,
            "complexity": 1, "call_count": 0, "literal_count": 0,
            "parse_error_kind": type(exc).__name__,
        }


def index_python_functions(repo: Path, source_package: str):
    """Index *production* Python function/method bodies from the buggy checkout."""
    if not source_package or Path(source_package).is_absolute() or ".." in Path(source_package).parts:
        raise ValueError("source_package must be a relative package directory")
    repo = repo.resolve()
    package = (repo / source_package).resolve()
    if not package.is_relative_to(repo) or not package.is_dir():
        raise FileNotFoundError(f"Production source package not found: {package}")
    indexed = {}
    for path in sorted(package.rglob("*.py")):
        if any(part in {"tests", "test", "testing", "__pycache__", ".git", "graphify-out"}
               for part in path.relative_to(repo).parts):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        lines = source.splitlines(keepends=True)
        rel = path.relative_to(repo).as_posix()
        def walk(body, parents=()):
            for n in body:
                if isinstance(n, ast.ClassDef):
                    ident = rel + "::" + ".".join((*parents, n.name))
                    import textwrap
                    start = min([n.lineno] + [d.lineno for d in n.decorator_list])
                    indexed[ident] = {
                        "source": textwrap.dedent("".join(lines[start - 1:n.end_lineno])),
                        "path": rel, "start": start, "end": n.end_lineno,
                        "doc": ast.get_docstring(n) or "",
                        "entity_type": "class",
                    }
                    walk(n.body, (*parents, n.name))
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ident = rel + "::" + ".".join((*parents, n.name))
                    # Full body, no artificial code truncation.
                    start = min([n.lineno] + [d.lineno for d in n.decorator_list])
                    end = n.end_lineno
                    segment = "".join(lines[start - 1:end])
                    # Top-level AST parsing needs dedentation for class methods.
                    import textwrap
                    indexed[ident] = {
                        "source": textwrap.dedent(segment),
                        "path": rel, "start": start, "end": end,
                        "doc": ast.get_docstring(n) or "",
                    }
                elif isinstance(n, (ast.If, ast.Try, ast.With)):
                    # Only module-level conditional definitions, no arbitrary
                    # function-body nested definitions as separate candidates.
                    for group in ("body", "orelse", "finalbody"):
                        walk(getattr(n, group, ()), parents)
        walk(tree.body)
    if not indexed:
        raise ValueError("No parseable production Python functions found")
    return indexed


def lexical_pool(indexed, context: str, limit=12):
    """Gold-free lexical source-retrieval baseline. Never calls patch labels."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    query = words(context)
    common = {"sympy", "python", "error", "test", "issue", "function", "please",
              "should", "could", "would", "when", "there", "with", "from"}
    query = query - common
    if not query:
        raise ValueError("No usable terms in observed issue/test evidence")
    scores = []
    for ident, item in indexed.items():
        # Production names and body *before* inference are legitimate retrieval features.
        full_words = words(ident.replace("::", " ") + " " + item["source"][:4000])
        label_words = words(ident + " " + item["doc"][:500])
        overlap = len(query & full_words)
        label_overlap = len(query & label_words)
        score = 2 * label_overlap + overlap
        scores.append((score, ident))
    scores.sort(key=lambda row: (-row[0], row[1]))
    if not scores or scores[0][0] <= 0:
        raise ValueError("Lexical search found no candidate overlapping failure evidence")
    top = scores[:limit]
    peak = max(1, scores[0][0])
    candidates = []
    sources = {}
    meta = {}
    for rank, (score, ident) in enumerate(top, 1):
        # Normalized rank proxy. Do not call it semantic similarity or coverage.
        prior = score / peak
        candidates.append(Candidate(ident, ident, semantic_score=prior))
        sources[ident] = indexed[ident]["source"]
        meta[ident] = {"prior": prior, "methods": ["lexical_source"],
                       "ranks": {"lexical_source": rank}, "top_per_method": len(top),
                       "source_file": indexed[ident]["path"],
                       "source_start": indexed[ident]["start"], "source_end": indexed[ident]["end"]}
    return candidates, sources, meta



def existing_pool(indexed: dict, pool_file: Path, condition="raw", limit=20):
    """Reuse the existing SWE-bench Agent4LR pool without gold access.

    The upstream file may have duplicate *positions*; verification requires
    unique functions. Preserve their first appearance and source provenance.
    Missing source is a hard error, not a replacement with a gold candidate.
    """
    content = json.loads(pool_file.read_text())
    rows = content.get(condition, {}).get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{pool_file}: missing {condition} candidate list")
    unique, seen, missing = [], set(), []
    for row in rows:
        ident = row.get("entity") if isinstance(row, dict) else None
        if not isinstance(ident, str) or not ident:
            raise ValueError("Malformed candidate-pool entity")
        if ident in seen:
            continue
        seen.add(ident)
        if ident not in indexed:
            missing.append(ident)
            continue
        unique.append((ident, row.get("source", "unknown")))
        if len(unique) >= limit:
            break
    if missing:
        raise ValueError(f"Candidate source absent from Python AST index ({len(missing)}): {missing[:5]}")
    if not unique:
        raise ValueError("No resolvable Python callable candidates in existing pool")
    candidates, sources, metadata = [], {}, {}
    for rank, (ident, origin) in enumerate(unique, 1):
        prior = 1 - (rank - 1) / max(1, len(unique))
        candidates.append(Candidate(ident, ident, semantic_score=prior))
        sources[ident] = indexed[ident]["source"]
        metadata[ident] = {"prior": prior, "methods": [str(origin)],
                           "ranks": {str(origin): rank}, "top_per_method": len(unique),
                           "source_file": indexed[ident]["path"],
                           "source_start": indexed[ident]["start"],
                           "source_end": indexed[ident]["end"]}
    return candidates, sources, metadata


def prepare_swebench(work: Path, source_package="sympy", top_k=12,
                     evidence_kind="raw", candidate_source="lexical",
                     pool_condition="raw"):
    repo = work / "repo"
    metadata_path = work / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing genuine SWE-bench metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    issue = metadata.get("problem_statement", "")
    if not isinstance(issue, str) or not issue.strip():
        raise ValueError("Missing SWE-bench problem_statement")
    tests = metadata.get("FAIL_TO_PASS", [])
    if not isinstance(tests, list):
        raise ValueError("FAIL_TO_PASS must be a list")
    if evidence_kind == "issue_only":
        # Official issue + failing test names, no manufactured test output.
        observed = ""
    else:
        output_filename = {"raw": "raw_test_output.txt", "leanctx": "leanctx_test_output.txt"}[evidence_kind]
        output = work / "outputs" / output_filename
        if not output.is_file():
            raise FileNotFoundError(f"Missing observed test log: {output}")
        observed = output.read_text(encoding="utf-8", errors="replace")
        if not observed.strip():
            raise ValueError(f"Observed {evidence_kind} output was empty")
    context = ("ISSUE:\n" + issue + "\n\nFAILING TESTS:\n" +
               "\n".join(map(str, tests)) + "\n\nOBSERVED TEST OUTPUT:\n" + observed[:12000])
    indexed = index_python_functions(repo, source_package)
    if candidate_source == "lexical":
        cs, sources, meta = lexical_pool(indexed, context, limit=top_k)
    elif candidate_source == "existing_pool":
        cs, sources, meta = existing_pool(
            indexed, work / "outputs" / "agent4lr" / "candidate_pools.json",
            condition=pool_condition, limit=top_k)
    else:
        raise ValueError("candidate_source must be lexical or existing_pool")
    # Reuse the shared RAVE-FL feature/risk pipeline, substituting Python AST
    # sketches only for this explicit SWE-bench runner. Never mutate global parser.
    import rave_fl.code_sketch as sketches
    original = sketches.java_sketch
    try:
        sketches.java_sketch = python_sketch
        prepared = prepare_candidates(cs, sources, meta, context)
    finally:
        sketches.java_sketch = original
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    return prepared, context, indexed, {
        "instance_id": metadata.get("instance_id", work.name),
        "dataset": "SWE-bench/Python", "candidate_source": candidate_source,
        "pool_condition": pool_condition if candidate_source == "existing_pool" else None,
        "gold_used_in_retrieval": False, "gold_used_in_prompt": False,
        "evidence_kind": evidence_kind, "has_observed_test_output": bool(observed),
        "source_package": source_package,
        "checkout_commit": revision or None, "candidate_count": len(prepared),
        "candidate_ids": [c.function_id for c in prepared],
        "metadata_keys_read_before_inference": ["instance_id", "problem_statement", "FAIL_TO_PASS"],
        "source_scores_are_measured_semantic_coverage": False,
        "source_hash": hashlib.sha256("".join(sources[i] for i in sources).encode()).hexdigest(),
        "evidence_sha256": hashlib.sha256(context.encode()).hexdigest(),
    }


def gold_from_patch(metadata_path: Path, indexed: dict):
    """Offline gold proxy: map actual changed old-side lines to *innermost* AST entity.

    IMPORTANT: additive class attributes are class-level gold, NOT an adjacent
    method merely because it appears inside a unified-diff context hunk.
    """
    metadata = json.loads(metadata_path.read_text())
    patch = metadata.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        return [], "patch_unavailable"
    labels, current_file, old_line = set(), None, None
    changed = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        hunk = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
        if hunk:
            old_line = int(hunk.group(1))
            continue
        if old_line is None or current_file is None:
            continue
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("-") and not line.startswith("---"):
            changed.append((current_file, old_line))
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            changed.append((current_file, old_line))
        elif line.startswith(" "):
            old_line += 1
    for filename, lineno in changed:
        possible = [(item["end"] - item["start"], ident)
                    for ident, item in indexed.items()
                    if item["path"] == filename and item["start"] <= lineno <= item["end"]]
        if possible:
            labels.add(min(possible)[1])
    return sorted(labels), "changed_line_innermost_ast_entity_proxy"


def score_ranking(ranking, labels):
    rank = next((i for i, ident in enumerate(ranking, 1) if ident in labels), None)
    return {"first_relevant_rank": rank, "top1": rank == 1, "top5": bool(rank and rank <= 5),
            "MRR": 1.0 / rank if rank else 0.0,
            "label_type": "changed_line_innermost_ast_entity_proxy",
            "candidate_gold_recall": any(c in labels for c in ranking)}


def run(args):
    work = args.work.resolve()
    candidates, context, indexed, provenance = prepare_swebench(
        work, args.source_package, args.top_k, args.evidence,
        getattr(args, "candidate_source", "lexical"), getattr(args, "pool_condition", "raw"))
    arms = args.arms.split(",")
    valid_arms = {"full", "fixed_batch", "confidence", "adaptive"} | REPAIR_MODES
    if not arms or len(arms) != len(set(arms)) or any(a not in valid_arms for a in arms):
        raise ValueError("Unsupported/duplicate arms")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; never overwrite research evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "starting", "provenance": provenance, "dry_run": args.dry_run,
              "model": args.model, "arms": {}, "limitations": [
                "Lexical source retrieval is not the original SieveFL localizer.",
                "Python AST sketches are lossy, input L2 is complete source.",
                "Patch-hunk mapping is a proxy, not official SWE-bench resolved-patch evaluation.",
                "Only post-localizer verification tokens are measured.",
                "One development bug cannot prove accuracy preservation.",
              ]}
    (args.output / "input.json").write_text(json.dumps({
        "provenance": provenance, "candidates": [asdict(c) for c in candidates],
        "context": context}, indent=2))
    for arm in arms:
        policy_budget = args.full_budget if arm == "full" else args.policy_budget
        backend = OllamaPilotBackend(args.model or "not_configured", context, args.base_url,
                                     args.max_output_tokens, args.timeout,
                                     ledger=args.output / f"{arm}_calls.jsonl",
                                     context_window=args.context_window, language="Python")
        if args.dry_run:
            result["arms"][arm] = {
                "llm_called": False,
                "estimated_full_tokens": sum(backend.estimate([c], Action.FULL_VERIFY, 2) for c in candidates),
                "budget": policy_budget,
            }
            continue
        backend.preflight()
        config = PolicyConfig(token_budget=policy_budget, batch_size=args.batch_size, mode=arm)
        engine = (RankRepairPilot(candidates, backend, config,
                                  RepairConfig(max_full=args.max_full_repairs))
                  if arm in REPAIR_MODES else AdaptivePilot(candidates, backend, config))
        started = time.monotonic()
        measured = engine.execute()
        measured["wall_seconds"] = time.monotonic() - started
        result["arms"][arm] = measured
        (args.output / f"{arm}.json").write_text(json.dumps(measured, indent=2))
        (args.output / "run.json").write_text(json.dumps(result, indent=2))
    if args.dry_run:
        result["status"] = "preflight_only"
        result["gold_read"] = False
    else:
        gold, label_type = gold_from_patch(work / "metadata.json", indexed)
        result["gold_label_type"] = label_type
        result["gold_read"] = True
        result["gold_entities"] = gold
        for arm, measured in result["arms"].items():
            measured["accuracy"] = score_ranking(measured["ranked_functions"], gold) if gold else None
            (args.output / f"{arm}.json").write_text(json.dumps(measured, indent=2))
        result["status"] = "complete"
    (args.output / "run.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-package", default="sympy")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--candidate-source", choices=("lexical", "existing_pool"), default="lexical",
                   help="For existing_pool require the genuine saved Agent4LR candidate_pools.json")
    p.add_argument("--pool-condition", choices=("raw", "leanctx"), default="raw")
    p.add_argument("--evidence", choices=("raw", "leanctx", "issue_only"), default="raw")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--arms", default="full,fixed_batch,repair_utility,repair_grounded")
    p.add_argument("--policy-budget", type=int, default=16000)
    p.add_argument("--full-budget", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-full-repairs", type=int, default=4)
    p.add_argument("--max-output-tokens", type=int, default=160)
    p.add_argument("--context-window", type=int, default=32768)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    data = run(args)
    print(json.dumps({"status": data["status"], "instance": data["provenance"]["instance_id"],
                      "candidate_count": data["provenance"]["candidate_count"],
                      "arms": {k: {m: v.get(m) for m in ("call_count","actual_tokens","accuracy","budget","estimated_full_tokens")}
                               for k,v in data["arms"].items()},
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
