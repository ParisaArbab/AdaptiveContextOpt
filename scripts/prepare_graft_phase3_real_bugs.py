"""Prepare authentic Defects4J buggy checkouts and full failing-test evidence.

This is a SEPARATE exploratory BM25 condition. It does not alter the frozen
FlexFL experiment or use fabricated failure text. Default mode only checks
availability. Materialization runs real Defects4J checkout/tests ONLY with
explicit flags and records exact provenance, including original SHA mismatch.

Run from the AdaptiveContextOpt repository root. No LLM is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from rave_fl.eval.representations import java_tree

BUGS = ("Time-1", "Lang-1", "Math-1")
FIXED = {"Time-1": ("Time", "1"), "Lang-1": ("Lang", "1"),
         "Math-1": ("Math", "1")}
_MAX_BYTES = 9000


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def cmd(args: list[str], *, cwd=None, timeout=1200) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True,
                          timeout=timeout, check=False)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("Expected a frozen manifest with cases list")
    return value


def _only_known_cases(cases):
    if [row.get("bug") for row in cases] != list(BUGS):
        raise ValueError("Cohort IDs/order must be Time-1, Lang-1, Math-1")
    return cases


def inspect(manifest: Path, cli: str | None, root: Path) -> dict:
    cases = _only_known_cases(_read_json(manifest)["cases"])
    source = shutil.which(cli) if cli else shutil.which("defects4j")
    if cli and Path(cli).is_file():
        source = str(Path(cli).resolve())
    details = []
    for case in cases:
        bug = case["bug"]
        checkout = root / "checkouts" / bug
        details.append({
            "bug": bug,
            "fixed_original_id": bug,
            "frozen_predeclared_revision": case.get("buggy_revision", ""),
            "official_d4j_checkout": str(checkout.resolve()),
            "checkout_exists": checkout.is_dir(),
            "source_evidence_available": False,
            "candidate_provider": "bm25",
            "gold_status": "not_checked",
        })
    return {
        "schema": "graft-phase3-official-d4j-preparation-v1",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": digest(manifest.read_bytes()),
        "d4j_executable": source,
        "d4j_available": bool(source),
        "condition": "official-Defects4J-buggy-checkout + observed-test + BM25",
        "not_equivalent_to_original_flexfl_condition": True,
        "network_or_tests_performed": False,
        "new_model_calls": 0,
        "bugs": details,
    }


def _one_test_method(source: bytes, name: str) -> bytes | None:
    """Return the exact raw method bytes from tree-sitter, not a regex slice."""
    tree = java_tree(source.decode("utf-8", errors="replace"))
    if tree.has_error:
        return None
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.type == "method_declaration":
            nm = node.child_by_field_name("name")
            if nm and source[nm.start_byte:nm.end_byte].decode(errors="replace") == name:
                return source[node.start_byte:node.end_byte]
        stack.extend(reversed(node.named_children))
    return None


def find_test_source(checkout: Path, identifier: str, test_directory: str) -> tuple[Path, bytes]:
    # The class/method comes only from observed Defects4J test metadata.
    if "::" not in identifier:
        raise ValueError("Need official Class::method failing-test identifier")
    klass, method = identifier.split("::", 1)
    if not re.fullmatch(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+", klass):
        raise ValueError("Unrecognized test class identifier")
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", method):
        raise ValueError("Unrecognized test method")
    expected = klass.replace(".", "/") + ".java"
    preferred = (checkout / test_directory / expected).resolve()
    allowed = checkout.resolve()
    if not preferred.is_relative_to(allowed):
        raise ValueError("Test path escapes versioned checkout")
    candidates = [preferred] if preferred.is_file() else list(checkout.rglob(Path(expected).name))
    candidates = [
        p for p in candidates
        if p.is_file() and not p.is_symlink()
        and str(p.resolve()).endswith(expected)
        and p.resolve().is_relative_to(allowed)
    ]
    if len(candidates) != 1:
        raise ValueError(f"Test method source missing/ambiguous: {identifier}, matches={len(candidates)}")
    raw = candidates[0].read_bytes()
    method_raw = _one_test_method(raw, method)
    if not method_raw:
        raise ValueError(f"No complete parsed method body for {identifier}")
    return candidates[0], method_raw


def _export(cli: str, property_name: str, checkout: Path) -> str:
    result = cmd([cli, "export", "-p", property_name, "-w", str(checkout)], timeout=60)
    if result.returncode:
        raise RuntimeError(f"Defects4J export {property_name} failed: {result.stderr[-400:]}")
    return result.stdout.strip()


def _git_revision(checkout: Path) -> tuple[str, str]:
    rev = cmd(["git", "-C", str(checkout), "rev-parse", "HEAD"], timeout=30)
    dirty = cmd(["git", "-C", str(checkout), "status", "--porcelain",
                 "--untracked-files=no"], timeout=30)
    if rev.returncode or dirty.returncode or len(rev.stdout.strip()) != 40:
        raise RuntimeError(f"Checkout is not a usable Git commit: {checkout}")
    return rev.stdout.strip(), dirty.stdout.strip()


def _observed_failing_tests(checkout: Path) -> list[str]:
    path = checkout / "failing_tests"
    if not path.is_file():
        raise RuntimeError(f"Defects4J did not produce failing_tests: {path}")
    records = re.findall(r"(?m)^---\s+([\w.$]+::[\w$]+)\s*$",
                         path.read_text(errors="replace"))
    return list(dict.fromkeys(records))


def _file_labels(checkout: Path, classes: str, source_dir: str) -> list[str]:
    root = checkout.resolve()
    labels = []
    for klass in [line.strip() for line in classes.splitlines() if line.strip()]:
        if not re.fullmatch(r"[\w$]+(?:\.[\w$]+)+", klass):
            continue
        path = root / source_dir / (klass.replace(".", "/") + ".java")
        if path.is_file() and path.resolve().is_relative_to(root):
            labels.append(path.relative_to(root).as_posix())
    return sorted(set(labels))


def collect_case(row: dict, cli: str, root: Path, *, max_bytes: int = _MAX_BYTES) -> tuple[dict, dict]:
    bug = row["bug"]
    project, number = FIXED[bug]
    checkout = root / "checkouts" / bug
    if checkout.exists():
        raise FileExistsError(f"Refusing to overwrite checkout: {checkout}")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    result = cmd([cli, "checkout", "-p", project, "-v", number + "b",
                  "-w", str(checkout)], timeout=1800)
    if result.returncode:
        raise RuntimeError(f"{bug} checkout failed: {result.stderr[-1500:]}")

    actual_sha, prior_dirty = _git_revision(checkout)
    if prior_dirty:
        raise RuntimeError(f"Buggy checkout is dirty before tests: {bug}")
    test_dir = _export(cli, "dir.src.tests", checkout)
    prod_dir = _export(cli, "dir.src.classes", checkout)
    trigger = _export(cli, "tests.trigger", checkout)
    # Real tests; never substitute an issue description or oracle patch.
    test_run = cmd([cli, "test", "-w", str(checkout)], timeout=1800)
    logs = root / "test_runs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{bug}.stdout.txt").write_text(test_run.stdout)
    (logs / f"{bug}.stderr.txt").write_text(test_run.stderr)
    fails = _observed_failing_tests(checkout)
    if not fails:
        raise RuntimeError(f"No confirmed failing test on official {bug} buggy version")
    if test_run.returncode:
        raise RuntimeError(f"{bug} Defects4J test command failed (rc={test_run.returncode}), logs saved")

    # Keep tests.trigger as metadata, not as proof of execution.
    declared = [t.strip() for t in trigger.splitlines() if "::" in t]
    selected = next((f for f in fails if f in declared), fails[0])
    test_path, method_raw = find_test_source(checkout, selected, test_dir)
    trace = (checkout / "failing_tests").read_bytes()
    if len(method_raw) > max_bytes or len(trace) > max_bytes:
        raise RuntimeError(
            f"{bug} complete method/trace too large for frozen bound, "
            f"method={len(method_raw)} trace={len(trace)}, cap={max_bytes}; no truncation"
        )
    revision_after, dirty_after = _git_revision(checkout)
    if revision_after != actual_sha or dirty_after:
        raise RuntimeError(f"{bug} checkout changed tracked source during tests; do not modify it")
    # Attach genuine full source plus observed failure trace with file hashes.
    evidence = (
        f"OFFICIAL DEFECTS4J BUG: {bug}\n"
        f"EXECUTED BUGGY TEST: {selected}\n"
        f"TEST SOURCE: {test_path.relative_to(checkout).as_posix()}\n"
        f"TEST SOURCE SHA256: {digest(test_path.read_bytes())}\n"
        "COMPLETE EXECUTED TEST METHOD (verbatim):\n"
    ).encode() + method_raw + (
        "\nOBSERVED DEFECTS4J FAILING_TESTS OUTPUT (verbatim):\n"
    ).encode() + trace
    if len(evidence) > 2 * max_bytes + 1000:
        raise RuntimeError("Evidence packaging exceeds sanity cap")
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / f"{bug}.txt"
    if evidence_path.exists():
        raise FileExistsError(evidence_path)
    evidence_path.write_bytes(evidence)
    # Derive only file-level modified-class proxy labels. We do not invent
    # method-level gold; this metadata is never in model input.
    classes = _export(cli, "classes.modified", checkout)
    file_labels = _file_labels(checkout, classes, prod_dir)
    if not file_labels:
        raise RuntimeError(
            f"{bug} official modified-class metadata produced no unique source file labels; "
            "do not claim measurable localization accuracy"
        )
    gold = {"files": file_labels, "functions": [],
            "semantics": "official_d4j_modified_classes_file_proxy"}
    case = {
        "bug": bug,
        "repository": row.get("source_repository", project),
        "buggy_revision": actual_sha,
        "checkout": str(checkout.resolve()),
        "evidence_file": str(evidence_path.resolve()),
        "evidence_kind": "observed_failure",
        "evidence_format": "text",
        "source_root": prod_dir,
        "gold": gold,
    }
    audit = {
        "bug": bug,
        "original_declared_sha": row.get("buggy_revision", ""),
        "actual_official_d4j_checkout_sha": actual_sha,
        "original_sha_matches": actual_sha == row.get("buggy_revision"),
        "test_method": selected,
        "test_method_path": str(test_path),
        "test_method_full_raw_sha256": digest(method_raw),
        "observed_trace_sha256": digest(trace),
        "evidence_sha256": digest(evidence),
        "evidence_bytes": len(evidence),
        "test_run_exit_code": test_run.returncode,
        "source_checkout_clean": True,
        "candidate_provider": "bm25",
        "modified_file_proxy_count": len(file_labels),
        "modified_file_proxy": file_labels,
        "method_gold_available": False,
        "fixture_synthetic": False,
    }
    return case, audit


def collect(manifest: Path, root: Path, cli: str, max_bytes: int = _MAX_BYTES) -> dict:
    data = _read_json(manifest)
    rows = _only_known_cases(data["cases"])
    if root.exists():
        raise FileExistsError(f"Preparation output must be new: {root}")
    root.mkdir(parents=True)
    complete, audits, failures = [], [], []
    for row in rows:
        try:
            case, audit = collect_case(row, cli, root, max_bytes=max_bytes)
            complete.append(case)
            audits.append(audit)
            print("READY:", row["bug"], "observed test", audit["test_method"],
                  "file proxy labels", audit["modified_file_proxy_count"])
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            failures.append({"bug": row["bug"], "error": str(exc)})
            print("UNAVAILABLE:", row["bug"], str(exc))
    output = {
        "schema": "graft-phase3-official-d4j-input-v1",
        "path_base": "cwd",
        "source_manifest_sha256": digest(manifest.read_bytes()),
        "condition": "official_buggy_checkout_observed_test_full_source_BM25",
        "not_original_flexfl_condition": True,
        "cases": complete,
    }
    # Do not create a runnable partially-replaced cohort.
    if len(complete) == len(BUGS):
        (root / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    report = {
        "cohort": list(BUGS),
        "status": "ready" if len(complete) == len(BUGS) else "incomplete",
        "instances_ready": [r["bug"] for r in complete],
        "issues": failures,
        "audits": audits,
        "new_model_calls": 0,
        "method_gold_available": False,
        "file_proxy_labels_only": True,
    }
    (root / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("configs/rave_independent_smoke.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--defects4j", default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--allow-checkout", action="store_true")
    ap.add_argument("--allow-real-tests", action="store_true")
    ap.add_argument("--max-evidence-bytes", type=int, default=_MAX_BYTES)
    args = ap.parse_args(argv)
    if args.max_evidence_bytes < 1000:
        raise SystemExit("At least 1000 complete evidence bytes required")
    current = inspect(args.manifest, args.defects4j, args.output)
    if not args.collect:
        print(json.dumps(current, indent=2))
        return 0
    if not args.allow_checkout or not args.allow_real_tests:
        raise SystemExit("Explicit --allow-checkout and --allow-real-tests required")
    if not current["d4j_executable"]:
        print("Defects4J not installed in PATH. See https://github.com/rjust/defects4j")
        return 3
    result = collect(args.manifest, args.output, current["d4j_executable"],
                     max_bytes=args.max_evidence_bytes)
    print(json.dumps({"status": result["status"], "issues": result["issues"]}, indent=2))
    return 0 if result["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
