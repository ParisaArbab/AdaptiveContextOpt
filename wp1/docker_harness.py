"""
docker_harness.py — WP1

Captures real pytest terminal output for a SWE-bench Lite instance.

Authoritative mode (unchanged from the original WP1 harness): run the
official SWE-bench Docker image for the instance and capture real stdout.
This sandbox has no Docker daemon (`docker: not found`), same "blocked on
Docker" constraint noted from earlier sessions — this module still targets
the real images; it just can't execute them here.

`--local-fallback` mode: clones the repo at base_commit and runs pytest
directly on the host, for validating the rest of the pipeline (graphify
step, compressor, feedback loop, scoring) without Docker. Output from this
mode is NOT equivalent to the official harness (no environment isolation,
dependency versions may drift) and must not be used for the formal
Compression Tax numbers — same rule that applied to the original mock mode.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestRunResult:
    instance_id: str
    stdout: str
    stderr: str
    returncode: int
    mode: str  # "docker" | "local_fallback"


SWEBENCH_IMAGE_TEMPLATE = "sweb.eval.x86_64.{instance_id_safe}:latest"


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def run_in_docker(instance_id: str, timeout: int = 900) -> TestRunResult:
    """Run the official SWE-bench eval image for this instance and capture
    real stdout/stderr. Requires the SWE-bench harness images to be built
    locally (see princeton-nlp/SWE-bench's `run_evaluation` docs) — this
    function shells out to `docker run` against that pre-built image rather
    than reimplementing SWE-bench's own harness."""
    if not _docker_available():
        raise RuntimeError(
            "Docker daemon not reachable. Build/pull SWE-bench eval images first "
            "(see princeton-nlp/SWE-bench docs), or use --local-fallback for "
            "pipeline validation only."
        )

    image = SWEBENCH_IMAGE_TEMPLATE.format(instance_id_safe=instance_id.replace("/", "_"))
    cmd = ["docker", "run", "--rm", image]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return TestRunResult(
        instance_id=instance_id,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        mode="docker",
    )


def run_local_fallback(
    instance_id: str,
    repo: str,
    base_commit: str,
    test_patch: str,
    workdir: Optional[Path] = None,
    timeout: int = 300,
) -> TestRunResult:
    """Clone `repo` at `base_commit`, apply `test_patch`, run pytest, capture
    real output. Explicitly NOT a substitute for the Docker path — flagged
    mode='local_fallback' everywhere downstream so it can be excluded from
    the formal Compression Tax report."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix=f"wp1_{instance_id.replace('/', '_')}_"))
    repo_url = f"https://github.com/{repo}.git"

    clone_cmd = ["git", "clone", repo_url, str(workdir)]
    clone = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=timeout)
    if clone.returncode != 0:
        return TestRunResult(instance_id, "", clone.stderr, clone.returncode, "local_fallback")

    subprocess.run(["git", "checkout", base_commit], cwd=workdir, capture_output=True, timeout=60)

    if test_patch:
        patch_file = workdir / "_wp1_test.patch"
        patch_file.write_text(test_patch)
        apply_result = subprocess.run(
            ["git", "apply", str(patch_file)], cwd=workdir, capture_output=True, text=True
        )
        if apply_result.returncode != 0:
            return TestRunResult(
                instance_id, "", f"test_patch apply failed: {apply_result.stderr}", 1, "local_fallback"
            )

    pytest_cmd = ["python3", "-m", "pytest", "-v", "--tb=long"]
    result = subprocess.run(
        pytest_cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout
    )
    return TestRunResult(
        instance_id=instance_id,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        mode="local_fallback",
    )


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=str, required=True, help="instance_id")
    ap.add_argument("--repo", type=str, help="owner/repo, required for --local-fallback")
    ap.add_argument("--base-commit", type=str)
    ap.add_argument("--test-patch-file", type=str, default=None)
    ap.add_argument("--local-fallback", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.local_fallback:
        test_patch = Path(args.test_patch_file).read_text() if args.test_patch_file else ""
        r = run_local_fallback(args.instance, args.repo, args.base_commit, test_patch)
    else:
        r = run_in_docker(args.instance)

    print(f"[{r.mode}] instance={r.instance_id} returncode={r.returncode} "
          f"stdout_chars={len(r.stdout)}")
    if args.out:
        Path(args.out).write_text(json.dumps(r.__dict__, indent=2))
