"""
docker_harness.py — WP1 step 1: capture the REAL trigger-test failure.

What was wrong before: the orchestrator called run_local_fallback with
test_patch="" and this module then ran a bare `pytest -v` over the whole
repo at base_commit. That meant (a) the FAIL_TO_PASS test wasn't even
present in the tree, since it ships in the test patch, and (b) the captured
text was a whole-suite run rather than the trigger test's failure. FlexFL's
Stage 1 is built on stack-trace/trigger-test evidence, so the input the
entire compression experiment operates on was largely missing the evidence
it was supposed to be compressing. Fixed here:

  * the gold test patch is always applied before running,
  * only the FAIL_TO_PASS ids are run, via the language adapter's own
    command builder (pytest node ids for Python, Maven/Gradle selectors for
    Java),
  * the checkout is reused across runs instead of re-cloned per instance,
  * a run that produced no failure evidence is flagged (`has_failure_evidence`)
    rather than silently feeding an empty string into the pipeline.

Two execution modes, unchanged in intent:
  docker         — the official SWE-bench eval image for the instance. This
                   is the authoritative mode; it runs the trigger tests
                   inside the image's own prepared environment.
  local_fallback — same commands on the host. Environment isn't isolated and
                   dependency versions can drift, so results carry
                   mode='local_fallback' and are excluded from headline
                   numbers by the analyzer.
"""
from __future__ import annotations

import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from benchmarks import LanguageAdapter, PythonAdapter


@dataclass
class TestRunResult:
    instance_id: str
    stdout: str
    stderr: str
    returncode: int
    mode: str  # "docker" | "local_fallback"
    command: str = ""
    test_patch_applied: bool = False
    trigger_tests: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.stdout + ("\n" + self.stderr if self.stderr else "")

    @property
    def has_failure_evidence(self) -> bool:
        """A capture with no failure signal can't support fault localization —
        it means the trigger test didn't run, not that the bug is absent."""
        blob = self.text
        markers = ("FAILED", "ERROR", "Traceback", "AssertionError",
                   "FAILURES", "Tests run:", "FAILED:", "expected:")
        return any(m in blob for m in markers)


SWEBENCH_IMAGE_TEMPLATE = "sweb.eval.x86_64.{instance_id_safe}:latest"

# The SWE-bench eval images ship the repo at /testbed inside a conda env
# named `testbed`; the tests won't resolve imports without activating it.
_DOCKER_PREAMBLE = (
    "source /opt/miniconda3/bin/activate 2>/dev/null || true; "
    "conda activate testbed 2>/dev/null || true; "
    "cd /testbed"
)


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def ensure_checkout(repo: str, base_commit: str, workdir: Path, timeout: int = 600) -> Path:
    """Clone `repo` at `base_commit` into workdir, reusing an existing
    checkout. Graphify needs real source on disk in BOTH modes (the graph is
    built from the host filesystem, not from inside the image), so this runs
    regardless of which execution mode is used for the tests.

    Any local modifications (a previously applied test patch) are reset, so
    the structure map is always built from the pristine buggy commit — a
    test patch left applied would put test-file entities into the structure
    map and quietly change what every condition can predict."""
    workdir = Path(workdir)
    if (workdir / ".git").exists():
        subprocess.run(["git", "checkout", "--force", base_commit], cwd=workdir,
                       capture_output=True, timeout=120)
        subprocess.run(["git", "clean", "-fdx", "-e", "graphify-out"], cwd=workdir,
                       capture_output=True, timeout=120)
        return workdir

    workdir.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(workdir)],
        capture_output=True, text=True, timeout=timeout,
    )
    if clone.returncode != 0:
        raise RuntimeError(f"git clone {repo} failed:\n{clone.stderr}")
    checkout = subprocess.run(["git", "checkout", "--force", base_commit], cwd=workdir,
                              capture_output=True, text=True, timeout=120)
    if checkout.returncode != 0:
        raise RuntimeError(f"git checkout {base_commit} failed:\n{checkout.stderr}")
    return workdir


def _apply_test_patch(workdir: Path, test_patch: str) -> tuple[bool, Optional[str]]:
    """Applies the gold test patch, which is where the FAIL_TO_PASS test
    lives. Without this the trigger test simply doesn't exist in the tree."""
    if not test_patch:
        return False, "no test_patch supplied"
    patch_file = Path(workdir) / "_wp1_test.patch"
    patch_file.write_text(test_patch)
    for args in (["git", "apply", "-v", str(patch_file)],
                 ["git", "apply", "-v", "--3way", str(patch_file)],
                 ["patch", "-p1", "-i", str(patch_file)]):
        result = subprocess.run(args, cwd=workdir, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, None
    return False, f"test_patch apply failed: {result.stderr[:400]}"


def run_local_fallback(
    instance_id: str,
    repo: str,
    base_commit: str,
    test_patch: str,
    fail_to_pass: Sequence[str],
    adapter: Optional[LanguageAdapter] = None,
    workdir: Optional[Path] = None,
    timeout: int = 900,
) -> TestRunResult:
    adapter = adapter or PythonAdapter()
    workdir = Path(workdir or tempfile.mkdtemp(prefix=f"wp1_{instance_id.replace('/', '_')}_"))
    notes: List[str] = []

    ensure_checkout(repo, base_commit, workdir)
    applied, note = _apply_test_patch(workdir, test_patch)
    if note:
        notes.append(note)

    trigger_tests = list(fail_to_pass)
    if not trigger_tests:
        notes.append("no FAIL_TO_PASS ids; running the adapter's default selection")
    cmd = adapter.build_test_command(workdir, trigger_tests)

    try:
        result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = f"TIMEOUT after {timeout}s"
        code = 124
        notes.append(stderr)
    except FileNotFoundError as e:
        stdout, stderr, code = "", f"test runner not found: {e}", 127
        notes.append(stderr)

    run = TestRunResult(
        instance_id=instance_id, stdout=stdout, stderr=stderr, returncode=code,
        mode="local_fallback", command=shlex.join(cmd), test_patch_applied=applied,
        trigger_tests=trigger_tests, notes=notes,
    )
    if not run.has_failure_evidence:
        run.notes.append(
            "no failure evidence in captured output — the trigger test likely "
            "never ran (missing deps, wrong runner, or unapplied test patch)"
        )
    return run


def run_in_docker(
    instance_id: str,
    test_patch: str,
    fail_to_pass: Sequence[str],
    adapter: Optional[LanguageAdapter] = None,
    timeout: int = 1800,
) -> TestRunResult:
    """Runs the trigger tests inside the official SWE-bench eval image.

    Previously this shelled out to `docker run --rm image` with no command,
    which starts the image's default entrypoint and runs no tests at all.
    Now it applies the test patch inside the container and invokes exactly
    the FAIL_TO_PASS ids via the language adapter's command builder."""
    adapter = adapter or PythonAdapter()
    if not _docker_available():
        raise RuntimeError(
            "Docker daemon not reachable. Build/pull the SWE-bench eval images "
            "(see princeton-nlp/SWE-bench docs), or pass --local-fallback for "
            "pipeline validation only."
        )

    image = SWEBENCH_IMAGE_TEMPLATE.format(instance_id_safe=instance_id.replace("/", "_"))
    trigger_tests = list(fail_to_pass)
    test_cmd = shlex.join(adapter.build_test_command(Path("/testbed"), trigger_tests))

    with tempfile.TemporaryDirectory() as tmp:
        patch_host = Path(tmp) / "test.patch"
        patch_host.write_text(test_patch or "")
        apply_step = (
            "git apply -v /wp1/test.patch || git apply -v --3way /wp1/test.patch || "
            "patch -p1 -i /wp1/test.patch"
        ) if test_patch else "true"
        script = f"{_DOCKER_PREAMBLE}; {apply_step}; {test_cmd}"
        cmd = ["docker", "run", "--rm", "-v", f"{tmp}:/wp1:ro", image, "/bin/bash", "-c", script]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
            notes: List[str] = []
        except subprocess.TimeoutExpired:
            stdout, stderr, code = "", f"TIMEOUT after {timeout}s", 124
            notes = [stderr]

    run = TestRunResult(
        instance_id=instance_id, stdout=stdout, stderr=stderr, returncode=code,
        mode="docker", command=test_cmd, test_patch_applied=bool(test_patch),
        trigger_tests=trigger_tests, notes=notes,
    )
    if not run.has_failure_evidence:
        run.notes.append("no failure evidence in captured output — check the image and test ids")
    return run


if __name__ == "__main__":
    import argparse
    import json

    from benchmarks import LANGUAGES

    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--base-commit")
    ap.add_argument("--test-patch-file", default=None)
    ap.add_argument("--fail-to-pass", default="", help="comma-separated test ids")
    ap.add_argument("--language", default="python", choices=sorted(LANGUAGES))
    ap.add_argument("--local-fallback", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    patch_text = Path(args.test_patch_file).read_text() if args.test_patch_file else ""
    tests = [t for t in args.fail_to_pass.split(",") if t.strip()]
    adapter = LANGUAGES[args.language]

    if args.local_fallback:
        r = run_local_fallback(args.instance, args.repo, args.base_commit, patch_text,
                               tests, adapter, Path(args.workdir) if args.workdir else None)
    else:
        r = run_in_docker(args.instance, patch_text, tests, adapter)

    print(f"[{r.mode}] instance={r.instance_id} rc={r.returncode} "
          f"patch_applied={r.test_patch_applied} evidence={r.has_failure_evidence} "
          f"chars={len(r.text)}")
    print(f"  cmd: {r.command}")
    for n in r.notes:
        print(f"  note: {n}")
    if args.out:
        Path(args.out).write_text(json.dumps(r.__dict__, indent=2))
