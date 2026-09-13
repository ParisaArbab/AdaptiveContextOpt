from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entity:
    file: str
    name: str
    start: int
    end: int
    kind: str

    # Lines representing actual runtime behavior of this entity.
    #
    # IMPORTANT:
    # Python executes `def` and `class` declaration lines while importing
    # a module. Those import-time lines must NOT count as execution of the
    # function/class for spectrum-based fault localization.
    executable_lines: tuple[int, ...] = ()

    @property
    def ref(self) -> str:
        return f"{self.file}::{self.name}"


def run(cmd, cwd: Path, timeout: int = 600):
    print("\n$", " ".join(str(x) for x in cmd), flush=True)

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
    )

    print(proc.stdout[-4000:], flush=True)

    return proc


def test_files_from_patch(patch: str) -> list[str]:
    files = []

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()

            if path.endswith(".py") and path not in files:
                files.append(path)

    return files


def collect_pytest_tests(repo: Path, test_files: list[str]) -> list[str]:
    if not test_files:
        return []

    proc = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            *test_files,
        ],
        repo,
    )

    tests = []

    for line in proc.stdout.splitlines():
        line = line.strip()

        if "::" not in line:
            continue

        if line.startswith("<"):
            continue

        if ".py::" not in line:
            continue

        if line not in tests:
            tests.append(line)

    return tests


def test_name(nodeid: str) -> str:
    tail = nodeid.split("::")[-1]

    # Remove pytest parameterization suffix:
    # test_x[param] -> test_x
    tail = tail.split("[", 1)[0]

    return tail.strip()


def _runtime_lines_for_function(node) -> tuple[int, ...]:
    """
    Return source lines belonging to the executable BODY of a function.

    The function declaration line itself is intentionally excluded because
    Python executes it when the module is imported.

    Nested function/class bodies are separate scopes and are not attributed
    to the enclosing function.
    """
    lines: set[int] = set()

    def visit(current):
        # A nested scope is not runtime behavior of the outer function.
        if current is not node and isinstance(
            current,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            # The nested declaration statement itself may execute when the
            # outer function runs, but its body does not. We skip it entirely
            # to avoid false behavioral coverage.
            return

        if isinstance(current, ast.stmt):
            lineno = getattr(current, "lineno", None)
            end_lineno = getattr(current, "end_lineno", lineno)

            if lineno is not None:
                for line in range(
                    int(lineno),
                    int(end_lineno or lineno) + 1,
                ):
                    lines.add(line)

        for child in ast.iter_child_nodes(current):
            visit(child)

    # Start from body, never from the FunctionDef node itself.
    for statement in node.body:
        visit(statement)

    return tuple(sorted(lines))


def _runtime_lines_for_class(node: ast.ClassDef) -> tuple[int, ...]:
    """
    Represent class runtime behavior using executable lines of its methods.

    We intentionally ignore class declaration and class-body definition lines,
    because they run during module import and caused the previous 10,862-way
    Ochiai tie.

    If any direct method of the class actually executes, the class can be
    considered behaviorally covered.
    """
    lines: set[int] = set()

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.update(_runtime_lines_for_function(child))

    return tuple(sorted(lines))


def collect_entities(
    repo: Path,
    source_package: str | None = None,
) -> list[Entity]:
    entities: list[Entity] = []

    source_prefix = ""

    if source_package:
        source_prefix = source_package.strip("/").replace("\\", "/") + "/"

    for path in repo.rglob("*.py"):
        try:
            rel = path.relative_to(repo).as_posix()
        except ValueError:
            continue

        # Restrict fault-localization candidates to production package code.
        # For sympy/sympy this means sympy/**, excluding repository tooling
        # such as top-level conftest.py.
        if source_prefix and not (
            rel == source_package.strip("/")
            or rel.startswith(source_prefix)
        ):
            continue

        # Tests are evidence, not production-code localization candidates.
        if "/tests/" in f"/{rel}/":
            continue

        if rel.startswith("graphify-out/"):
            continue

        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source)
        except Exception:
            continue

        def walk(node, class_stack):
            if isinstance(node, ast.ClassDef):
                qual = ".".join(class_stack + [node.name])

                runtime_lines = _runtime_lines_for_class(node)

                entities.append(
                    Entity(
                        file=rel,
                        name=qual,
                        start=int(node.lineno),
                        end=int(
                            getattr(
                                node,
                                "end_lineno",
                                node.lineno,
                            )
                        ),
                        kind="class",
                        executable_lines=runtime_lines,
                    )
                )

                new_stack = class_stack + [node.name]

                for child in node.body:
                    walk(child, new_stack)

                return

            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                qual = ".".join(class_stack + [node.name])

                runtime_lines = _runtime_lines_for_function(node)

                entities.append(
                    Entity(
                        file=rel,
                        name=qual,
                        start=int(node.lineno),
                        end=int(
                            getattr(
                                node,
                                "end_lineno",
                                node.lineno,
                            )
                        ),
                        kind="function",
                        executable_lines=runtime_lines,
                    )
                )

                # Nested functions remain separate entities.
                for child in node.body:
                    walk(child, class_stack + [node.name])

                return

            for child in ast.iter_child_nodes(node):
                walk(child, class_stack)

        walk(tree, [])

    unique = {}

    for entity in entities:
        unique[
            (
                entity.file,
                entity.name,
                entity.start,
                entity.end,
                entity.kind,
            )
        ] = entity

    return list(unique.values())


def coverage_for_test(
    repo: Path,
    nodeid: str,
    source_package: str,
    temp_json: Path,
) -> tuple[int, dict[str, set[int]]]:

    run(
        [sys.executable, "-m", "coverage", "erase"],
        repo,
    )

    proc = run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--source={source_package}",
            "-m",
            "pytest",
            nodeid,
            "-q",
        ],
        repo,
        timeout=900,
    )

    if not (repo / ".coverage").exists():
        return proc.returncode, {}

    export = run(
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            "-o",
            str(temp_json),
        ],
        repo,
    )

    if export.returncode != 0 or not temp_json.exists():
        return proc.returncode, {}

    data = json.loads(temp_json.read_text())

    covered = {}

    for path, info in data.get("files", {}).items():
        normalized = path.replace("\\", "/")

        # coverage may return absolute paths.
        try:
            p = Path(path)
            if p.is_absolute():
                normalized = p.resolve().relative_to(repo.resolve()).as_posix()
        except Exception:
            pass

        covered[normalized] = set(
            int(x)
            for x in info.get("executed_lines", [])
        )

    return proc.returncode, covered


def entity_covered(
    entity: Entity,
    coverage: dict[str, set[int]],
) -> bool:
    """
    True only when actual executable BODY lines of the entity were executed.

    Do not use entity.start..entity.end here. Doing that treats import-time
    `def`/`class` execution as if every function/class had run.
    """

    if not entity.executable_lines:
        return False

    lines = coverage.get(entity.file)

    if not lines:
        # Handle absolute/alternate names returned by coverage.py.
        for path, candidate_lines in coverage.items():
            normalized = path.replace("\\", "/")

            if normalized.endswith(entity.file):
                lines = candidate_lines
                break

    if not lines:
        return False

    return bool(
        set(entity.executable_lines).intersection(lines)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--instance", required=True)
    ap.add_argument("--root", default=str(Path.home() / "AdaptiveContextOpt"))
    ap.add_argument("--source-package", default=None)

    args = ap.parse_args()

    root = Path(args.root)
    work = root / "data" / "swebench_workspaces" / args.instance
    repo = work / "repo"
    out = work / "outputs" / "ochiai"

    out.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (work / "metadata.json").read_text()
    )

    repo_name = metadata.get("repo", "")

    source_package = args.source_package

    if not source_package:
        # sympy/sympy -> sympy
        source_package = repo_name.split("/")[-1]

    fail_names = {
        str(x).split("[", 1)[0]
        for x in metadata.get("FAIL_TO_PASS", [])
    }

    pass_names = {
        str(x).split("[", 1)[0]
        for x in metadata.get("PASS_TO_PASS", [])
    }

    patch = (work / "test_patch.diff").read_text(errors="replace")

    test_files = test_files_from_patch(patch)

    print("INSTANCE:", args.instance)
    print("SOURCE PACKAGE:", source_package)
    print("TEST FILES:", test_files)
    print("FAIL_TO_PASS:", sorted(fail_names))
    print("PASS_TO_PASS:", sorted(pass_names))

    tests = collect_pytest_tests(repo, test_files)

    print("\nCOLLECTED TESTS:", len(tests))

    for t in tests:
        print(" ", t)

    failing_nodes = [
        t for t in tests
        if test_name(t) in fail_names
    ]

    if pass_names:
        passing_nodes = [
            t for t in tests
            if test_name(t) in pass_names
        ]

        passing_source = "SWE-bench PASS_TO_PASS"

    else:
        # For instances where SWE-bench does not provide PASS_TO_PASS
        # names for this file, use the other passing tests in the same
        # patched test file as the spectrum.
        passing_nodes = [
            t for t in tests
            if t not in failing_nodes
        ]

        passing_source = "other tests in patched SWE-bench test file"

    print("\nFAILING NODES:", len(failing_nodes))

    for t in failing_nodes:
        print(" ", t)

    print("\nPASSING CANDIDATE NODES:", len(passing_nodes))
    print("PASSING SOURCE:", passing_source)

    entities = collect_entities(repo, source_package)

    print("\nPRODUCTION ENTITIES:", len(entities))

    spectra = []

    # ----------------------------------------
    # Failing tests
    # ----------------------------------------

    for idx, nodeid in enumerate(failing_nodes, 1):
        print(
            f"\n===== FAILING COVERAGE {idx}/{len(failing_nodes)} =====",
            flush=True,
        )

        temp = out / f"coverage_failing_{idx}.json"

        exit_code, cov = coverage_for_test(
            repo,
            nodeid,
            source_package,
            temp,
        )

        spectra.append(
            {
                "nodeid": nodeid,
                "expected": "fail",
                "actual_exit": exit_code,
                "coverage": cov,
            }
        )

    # ----------------------------------------
    # Passing tests
    # ----------------------------------------

    accepted_pass = 0

    for idx, nodeid in enumerate(passing_nodes, 1):
        print(
            f"\n===== PASSING COVERAGE {idx}/{len(passing_nodes)} =====",
            flush=True,
        )

        temp = out / f"coverage_passing_{idx}.json"

        exit_code, cov = coverage_for_test(
            repo,
            nodeid,
            source_package,
            temp,
        )

        # Ochiai passing spectrum should only contain tests
        # that actually pass on the buggy commit.
        if exit_code == 0:
            accepted_pass += 1

            spectra.append(
                {
                    "nodeid": nodeid,
                    "expected": "pass",
                    "actual_exit": exit_code,
                    "coverage": cov,
                }
            )
        else:
            print(
                f"SKIP passing candidate because it failed: {nodeid}",
                flush=True,
            )

    total_failed = sum(
        1
        for x in spectra
        if x["expected"] == "fail"
    )

    total_passed = sum(
        1
        for x in spectra
        if x["expected"] == "pass"
    )

    print("\nTOTAL FAILING SPECTRA:", total_failed)
    print("TOTAL PASSING SPECTRA:", total_passed)

    rows = []

    for entity in entities:
        failed_covered = 0
        passed_covered = 0

        for spectrum in spectra:
            covered = entity_covered(
                entity,
                spectrum["coverage"],
            )

            if not covered:
                continue

            if spectrum["expected"] == "fail":
                failed_covered += 1
            else:
                passed_covered += 1

        if failed_covered == 0:
            continue

        denominator = math.sqrt(
            total_failed
            * (failed_covered + passed_covered)
        )

        score = (
            failed_covered / denominator
            if denominator
            else 0.0
        )

        rows.append(
            {
                "entity": entity.ref,
                "file": entity.file,
                "name": entity.name,
                "kind": entity.kind,
                "start": entity.start,
                "end": entity.end,
                "failed_covered": failed_covered,
                "passed_covered": passed_covered,
                "score": score,
            }
        )

    rows.sort(
        key=lambda x: (
            -x["score"],
            -x["failed_covered"],
            x["passed_covered"],
            x["entity"],
        )
    )

    # Deduplicate candidate refs while preserving rank.
    ranked = []
    seen = set()

    for row in rows:
        if row["entity"] in seen:
            continue

        seen.add(row["entity"])
        ranked.append(row)

    result = {
        "instance_id": args.instance,
        "method": "Ochiai",
        "adaptation": "Python coverage.py spectrum",
        "passing_spectrum_source": passing_source,
        "failing_tests": [
            x["nodeid"]
            for x in spectra
            if x["expected"] == "fail"
        ],
        "passing_tests": [
            x["nodeid"]
            for x in spectra
            if x["expected"] == "pass"
        ],
        "total_failed": total_failed,
        "total_passed": total_passed,
        "top20": ranked[:20],
        "top5": [
            x["entity"]
            for x in ranked[:5]
        ],
    }

    (out / "ochiai.json").write_text(
        json.dumps(result, indent=2)
    )

    with (out / "ochiai.csv").open("w", newline="") as f:
        fields = [
            "rank",
            "entity",
            "kind",
            "file",
            "start",
            "end",
            "failed_covered",
            "passed_covered",
            "score",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for i, row in enumerate(ranked, 1):
            writer.writerow(
                {
                    "rank": i,
                    **{
                        k: row[k]
                        for k in fields
                        if k != "rank"
                    },
                }
            )

    print("\n==================================================")
    print("OCHIAI TOP-20")
    print("==================================================")

    for i, row in enumerate(ranked[:20], 1):
        print(
            f"{i:2}. {row['entity']}"
            f"  score={row['score']:.6f}"
            f"  fail={row['failed_covered']}"
            f"  pass={row['passed_covered']}"
        )

    print("\n==================================================")
    print("OCHIAI TOP-5")
    print("==================================================")

    for i, entity in enumerate(result["top5"], 1):
        print(f"Top-{i}: {entity}")

    print("\nSaved:")
    print(out / "ochiai.json")
    print(out / "ochiai.csv")


if __name__ == "__main__":
    main()
