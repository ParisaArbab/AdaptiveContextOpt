"""Defects4J checkout and test-output capture for the WP1 experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
import time
from typing import Sequence


@dataclass
class CommandCapture:
    command: list[str]
    cwd: str
    returncode: int
    output: str
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command is not installed or not on PATH: {name}")
    return path


def parse_bug_id(bug: str) -> tuple[str, int]:
    """Convert FlexFL-style IDs such as ``Time-25`` to Defects4J coordinates."""
    if "-" not in bug:
        raise ValueError(f"Expected PROJECT-ID, for example Time-25, got {bug!r}")
    project, raw_id = bug.rsplit("-", 1)
    try:
        bug_id = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"Invalid Defects4J bug id: {bug!r}") from exc
    return project, bug_id


def _run(
    command: Sequence[str],
    cwd: Path | None = None,
    timeout: int = 1200,
    check: bool = False,
) -> CommandCapture:
    started = time.perf_counter()
    proc = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    capture = CommandCapture(
        command=list(command),
        cwd=str(cwd or Path.cwd()),
        returncode=proc.returncode,
        output=proc.stdout or "",
        elapsed_seconds=time.perf_counter() - started,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(command)}\n"
            f"{capture.output[-8000:]}"
        )
    return capture


def checkout_bug(bug: str, work_root: Path, reuse: bool = True) -> Path:
    """Checkout the buggy Defects4J version, for example Time-25 -> Time 25b."""
    require_binary("defects4j")
    project, bug_id = parse_bug_id(bug)
    work_root = Path(work_root)
    repo = work_root / bug

    if reuse and (repo / ".defects4j.config").exists():
        return repo

    if repo.exists():
        shutil.rmtree(repo)
    repo.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["defects4j", "checkout", "-p", project, "-v", f"{bug_id}b", "-w", str(repo)],
        timeout=1800,
        check=True,
    )
    return repo


def export_property(repo: Path, property_name: str) -> str:
    """Read a Defects4J metadata property. Empty output is returned on failure."""
    capture = _run(
        ["defects4j", "export", "-p", property_name],
        cwd=repo,
        timeout=120,
        check=False,
    )
    return capture.output.strip() if capture.returncode == 0 else ""


def run_tests_once(repo: Path, timeout: int = 1800) -> CommandCapture:
    """Run the buggy program once and capture the exact stdout/stderr used by both arms.

    A non-zero return code is expected when tests fail, therefore this function never
    treats it as a harness error.
    """
    require_binary("defects4j")
    return _run(["defects4j", "test"], cwd=repo, timeout=timeout, check=False)
