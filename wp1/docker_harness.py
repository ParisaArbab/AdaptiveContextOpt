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

import re
import os
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
    coverage_json: Optional[dict] = None
    coverage_error: str = ""
    python_exe: Optional[str] = None
    test_command_argv: List[str] = field(default_factory=list)

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
        # `.wp1venv*` must survive the clean. Without the exclusion, the
        # per-instance venv that _ensure_dependencies_installed builds is
        # deleted before every run and rebuilt from scratch — its documented
        # "reuse across repeated runs" never actually happened, and each arm
        # paid a full dependency install. `graphify-out` is excluded for the
        # same reason: the extraction is cached per repo.
        subprocess.run(["git", "clean", "-fdx", "-e", "graphify-out", "-e", ".wp1venv*"],
                       cwd=workdir, capture_output=True, timeout=120)
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


def _ensure_dependencies_installed(
    workdir: Path, adapter: LanguageAdapter, timeout: int = 1800
) -> tuple[Optional[Path], List[str]]:
    """Installs the checked-out package plus a test runner into a dedicated
    per-instance virtualenv, and returns its python executable.

    This was the actual root cause behind every 'no failure evidence'
    result: the checkout was real and the test patch applied cleanly, but
    nothing ever ran `pip install -e .` or `pip install pytest`, so pytest
    couldn't even import the package (confirmed manually: `import sympy`
    failed with 'SymPy now depends on mpmath as an external library' before
    this fix, and 'No module named pytest' before that).

    A dedicated venv per instance — not the shared conda env — because
    consecutive instances from different repos (sympy, django, ...) install
    conflicting dependency versions into the same environment otherwise;
    each instance needing its own isolated install is exactly why the
    official SWE-bench harness uses one container per instance. This is the
    closest a host without Docker can get to that isolation.

    IMPORTANT CAVEAT, found while testing this fix: the whole-system python3
    (often 3.11+ on a modern host) is frequently too NEW for older SWE-bench
    Lite instances. Confirmed directly: sympy 1.5.dev (2019) imports
    `distutils`, which Python 3.12 removed entirely — `pip install -e .`
    succeeds, but `import sympy` still fails on a 3.12 interpreter. The
    original SWE-bench harness handles this with a per-instance,
    per-repo-version conda spec baked into its Docker images; the
    currently-installed `swebench` PyPI package no longer even ships that
    mapping as an importable dict (checked: `swebench.harness.constants` has
    been restructured and no longer exposes MAP_REPO_VERSION_TO_SPECS or
    equivalent — modern swebench fully delegates environment fidelity to
    pre-built Docker images, which is also why `run_in_docker` above is the
    authoritative path).

    Without that mapping, this function does the best a Docker-less host
    can: probes a short, honestly-a-heuristic list of Python versions via
    conda (broadly covering the era most SWE-bench Lite repos predate),
    stopping at the first one where both install AND import succeed. This
    is NOT equivalent to the real per-instance environment and is still
    local_fallback-only — but it resolves the specific distutils-era failure
    class instead of silently producing 'no failure evidence' for every
    older instance.

    Only runs for Python; Java's build systems (Maven/Gradle) manage their
    own dependency resolution and don't need this step.
    """
    notes: List[str] = []
    if adapter.name != "python":
        return None, notes

    venv_dir = workdir / ".wp1venv"
    venv_python = venv_dir / "bin" / "python3"
    if venv_python.exists():
        return venv_python, notes  # reuse across repeated runs of the same instance

    def _try_install(python_exe: Path) -> tuple[bool, str]:
        pip = [str(python_exe), "-m", "pip"]
        # Upgrading pip is a convenience, not a requirement, so it must not be
        # able to take the instance down. On a machine without network this
        # call blocks until its timeout and the raised TimeoutExpired
        # propagated all the way out, skipping the instance entirely — a
        # five-minute stall followed by a lost data point, for a step whose
        # failure does not matter.
        try:
            subprocess.run(pip + ["install", "--upgrade", "pip"], cwd=workdir,
                           capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            notes.append("pip self-upgrade timed out (offline?); continuing with the "
                         "existing pip")

        # Build-toolchain ladder. SWE-bench instances are pinned to commits
        # from 2019-2022, and modern build tooling breaks them in two
        # specific, reproducible ways:
        #
        #   setuptools >= 74 removed `setuptools.dep_util`. astropy's
        #   setup.py (through extension-helpers) imports `newer_group` from
        #   it, so `pip install -e .` dies with ModuleNotFoundError before
        #   any of the package's own code runs. Seen live on
        #   astropy__astropy-12907, in every Python-version probe, which is
        #   why the version ladder alone never rescued it — the interpreter
        #   was never the problem.
        #
        #   setuptools >= 60 vendors its own distutils by default, which
        #   breaks packages that reach into stdlib distutils internals.
        #   SETUPTOOLS_USE_DISTUTILS=stdlib restores the old behaviour.
        #
        # Pinning setuptools in the venv is not enough on its own: PEP 517
        # build isolation gives the build its own fresh environment with the
        # newest setuptools regardless of what the venv holds. The error
        # surfaces at "Getting requirements to build editable", which is the
        # isolated build, so the pinned rungs must also pass
        # --no-build-isolation for the pin to have any effect.
        toolchains = [
            ("modern", [], {}),
            ("setuptools<74", ["setuptools<74", "wheel"], {}),
            ("setuptools<60 + stdlib distutils", ["setuptools<60", "wheel"],
             {"SETUPTOOLS_USE_DISTUTILS": "stdlib"}),
        ]

        attempts: List[str] = []
        for label, pins, extra_env in toolchains:
            env = {**os.environ, **extra_env}
            if pins:
                pin_result = subprocess.run(
                    pip + ["install", *pins], cwd=workdir, env=env,
                    capture_output=True, text=True, timeout=300,
                )
                if pin_result.returncode != 0:
                    attempts.append(f"{label}: could not pin build tools")
                    continue
            isolation = [] if not pins else ["--no-build-isolation"]
            for target in ([".[test]"], [".[dev]"], ["."]):
                result = subprocess.run(
                    pip + ["install", "-e", *isolation, *target],
                    cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout,
                )
                if result.returncode == 0:
                    if label != "modern":
                        notes.append(f"installed with pinned build toolchain ({label})")
                    break
            else:
                attempts.append(f"{label}: {result.stderr.strip()[-200:]}")
                continue
            break
        else:
            return False, "pip install -e . failed for every build toolchain: " + " | ".join(attempts)

        pytest_install = subprocess.run(pip + ["install", "pytest"], cwd=workdir,
                                         capture_output=True, text=True, timeout=300)
        if pytest_install.returncode != 0:
            return False, f"pytest install failed: {pytest_install.stderr[-300:]}"

        # Verify via pytest's own conftest-loading path, not a bare `import
        # <package>` — confirmed these differ: a plain `import sympy`
        # succeeded here while pytest's conftest.py still hit
        # 'ModuleNotFoundError: No module named distutils', because
        # conftest's import chain reaches sympy.external.importtools (which
        # imports distutils) while a top-level `import sympy` alone doesn't.
        # --collect-only forces the same conftest chain pytest actually uses,
        # without running any real tests.
        collect = subprocess.run(
            [str(python_exe), "-m", "pytest", "--collect-only", "-q"],
            cwd=workdir, capture_output=True, text=True, timeout=180,
        )
        collect_text = collect.stdout + collect.stderr
        if "ModuleNotFoundError" in collect_text or "ImportError while loading conftest" in collect_text:
            return False, f"pytest --collect-only hit an import error post-install: {collect_text[-400:]}"
        return True, ""

    # 1) try the plain system python3 first — cheapest, and correct for
    #    instances new enough not to need an older interpreter
    create = subprocess.run(["python3", "-m", "venv", str(venv_dir)], cwd=workdir,
                             capture_output=True, text=True, timeout=120)
    if create.returncode == 0:
        ok, err = _try_install(venv_python)
        if ok:
            return venv_python, notes
        notes.append(f"system python3 venv failed post-install checks: {err}")
        subprocess.run(["rm", "-rf", str(venv_dir)], capture_output=True)

    # 2) fall back to a short probe of older interpreters via conda, if
    #    available — this is the heuristic described above, not a real spec
    conda_exe = subprocess.run(["which", "conda"], capture_output=True, text=True).stdout.strip()
    if not conda_exe:
        notes.append("conda not found; cannot probe alternate Python versions")
        return None, notes

    for py_version in ("3.8", "3.9", "3.6"):
        candidate_dir = workdir / f".wp1venv_py{py_version.replace('.', '')}"
        candidate_python = candidate_dir / "bin" / "python3"
        make = subprocess.run(
            ["conda", "create", "-y", "-p", str(candidate_dir), f"python={py_version}"],
            capture_output=True, text=True, timeout=timeout,
        )
        if make.returncode != 0 or not candidate_python.exists():
            notes.append(f"conda env for python={py_version} failed to create; skipping")
            continue
        ok, err = _try_install(candidate_python)
        if ok:
            notes.append(f"resolved via conda python={py_version} (heuristic probe, not the "
                          f"instance's real recorded environment)")
            return candidate_python, notes
        notes.append(f"python={py_version} failed post-install checks: {err}")
        subprocess.run(["rm", "-rf", str(candidate_dir)], capture_output=True)

    notes.append("exhausted all Python-version probes; falling back to system python3 unverified")
    return None, notes


def _resolve_bare_test_ids(test_patch: str, fail_to_pass: Sequence[str]) -> List[str]:
    """Some SWE-bench datasets record FAIL_TO_PASS as bare test names
    ('test__TR56') rather than pytest node ids ('path/to/file.py::test__TR56').
    Confirmed directly against sympy__sympy-17139: passing a bare name as a
    positional pytest argument doesn't resolve to a specific test — pytest
    falls back to collecting the whole rootdir, which is what surfaced the
    conftest import chain instead of the actual targeted test result.

    Resolves each bare name to its real file by searching the SAME gold
    test_patch diff for a matching `def <name>(` on an added line — mirrors
    the approach fetch_instances.py already uses for gold-patch symbol
    extraction from a unified diff, applied here to the test patch instead
    of the fix patch.
    """
    resolved = []
    file_segments: List[tuple[str, str]] = []
    current_file, buf = None, []
    for line in test_patch.splitlines():
        if line.startswith("+++ b/"):
            if current_file is not None:
                file_segments.append((current_file, "\n".join(buf)))
            current_file, buf = line[6:].strip(), []
        elif current_file is not None:
            buf.append(line)
    if current_file is not None:
        file_segments.append((current_file, "\n".join(buf)))

    for test_id in fail_to_pass:
        if "::" in test_id or "/" in test_id:
            resolved.append(test_id)  # already a real node id, leave as-is
            continue
        base_name = re.sub(r"\[.*\]$", "", test_id)  # strip parametrize suffix
        match = None
        for file_path, body in file_segments:
            # Matches whether the test is newly added (a '+' line) or an
            # existing test being modified (where 'def test_x():' only
            # appears as unified-diff hunk-header context, e.g.
            # '@@ -76,6 +76,10 @@ def test__TR56():', never as an added line
            # itself) — confirmed necessary against sympy__sympy-17139,
            # where test__TR56 is exactly this second case.
            if re.search(rf"def\s+{re.escape(base_name)}\s*\(", body):
                match = file_path
                break
        resolved.append(f"{match}::{test_id}" if match else test_id)
    return resolved


def run_local_fallback(
    instance_id: str,
    repo: str,
    base_commit: str,
    test_patch: str,
    fail_to_pass: Sequence[str],
    adapter: Optional[LanguageAdapter] = None,
    workdir: Optional[Path] = None,
    timeout: int = 900,
    pass_to_pass: Sequence[str] = (),
    collect_coverage: bool = True,
    install_timeout: int = 1800,
) -> TestRunResult:
    adapter = adapter or PythonAdapter()
    workdir = Path(workdir or tempfile.mkdtemp(prefix=f"wp1_{instance_id.replace('/', '_')}_"))
    notes: List[str] = []

    ensure_checkout(repo, base_commit, workdir)

    venv_python, install_notes = _ensure_dependencies_installed(workdir, adapter, install_timeout)
    notes.extend(install_notes)

    applied, note = _apply_test_patch(workdir, test_patch)
    if note:
        notes.append(note)

    trigger_tests = list(fail_to_pass)
    if not trigger_tests:
        notes.append("no FAIL_TO_PASS ids; running the adapter's default selection")
    elif adapter.name == "python":
        resolved = _resolve_bare_test_ids(test_patch, trigger_tests)
        unresolved = [t for t, r in zip(trigger_tests, resolved) if t == r and "::" not in r]
        if unresolved:
            notes.append(f"could not resolve file path for bare test id(s): {unresolved}; "
                         f"passed through as-is, pytest may not target them correctly")
        trigger_tests = resolved
    cmd = adapter.build_test_command(workdir, trigger_tests)
    if venv_python and cmd and cmd[0] == "python3":
        cmd[0] = str(venv_python)

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
        python_exe=str(venv_python) if venv_python else None,
        test_command_argv=list(cmd),
    )
    if not run.has_failure_evidence:
        run.notes.append(
            "no failure evidence in captured output — the trigger test likely "
            "never ran (missing deps, wrong runner, or unapplied test patch)"
        )

    # Ochiai's spectrum. Only worth collecting when the tests actually ran —
    # coverage of a run that never reached the trigger test tells us nothing
    # about which method is suspicious.
    if collect_coverage and run.has_failure_evidence:
        run.coverage_json, run.coverage_error = collect_spectrum(
            workdir, adapter, venv_python, trigger_tests, pass_to_pass,
            timeout=timeout,
        )
        if run.coverage_error:
            run.notes.append(f"coverage/Ochiai unavailable: {run.coverage_error}")
    elif collect_coverage:
        run.coverage_error = "skipped: the evidence capture produced no failure"

    return run


def collect_spectrum(
    workdir: Path,
    adapter: LanguageAdapter,
    python_exe: Optional[Path],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    max_passing: int = 40,
    timeout: int = 1800,
) -> tuple[Optional[dict], str]:
    """Second test run, under coverage, for Ochiai's spectrum.

    Deliberately a separate run from the evidence capture: the capture has
    to produce exactly the text the compression experiment operates on, and
    running it under coverage changes both the output and the timing. This
    run's output is thrown away — only the coverage data matters.

    Passing tests are what make Ochiai discriminating. With failing tests
    alone every executed method scores identically and the ranking is just
    "what ran", so PASS_TO_PASS is sampled here too, bounded by max_passing
    because some instances list thousands and the spectrum saturates long
    before that.
    """
    if python_exe is None:
        return None, "no verified interpreter; coverage would not import the package"
    try:
        import traditional_fl
    except ImportError as e:
        return None, f"traditional_fl unavailable: {e}"

    selected = list(fail_to_pass) + list(pass_to_pass)[:max_passing]
    if not selected:
        return None, "no tests to profile"

    cmd = adapter.build_test_command(workdir, selected)
    if cmd and cmd[0] == "python3":
        cmd[0] = str(python_exe)
    return traditional_fl.run_coverage(workdir, python_exe, cmd, timeout=timeout)


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
    if adapter.name == "python" and trigger_tests:
        trigger_tests = _resolve_bare_test_ids(test_patch, trigger_tests)
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
