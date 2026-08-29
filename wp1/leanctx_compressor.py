"""Use the real LeanCTX shell compressor on a captured Defects4J test output.

The experiment captures ``defects4j test`` once. LeanCTX ``ctx_compare`` then
runs the production shell-compression pipeline against those exact bytes. The
RAW and LeanCTX arms therefore start from identical test output.

There is intentionally no Python fallback. If LeanCTX is missing or fails, the
LeanCTX arm fails loudly instead of silently changing the experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


@dataclass
class LeanCtxResult:
    text: str
    original_tokens: int | None
    compressed_tokens: int | None
    saved_percent: float | None
    original_bytes: int
    compressed_bytes: int
    report: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("text", None)
        data.pop("report", None)
        return data


def compress_captured_shell_output(
    raw_output: str,
    project_root: Path,
    command: str = "defects4j test",
    binary: str = "lean-ctx",
    timeout: int = 180,
) -> LeanCtxResult:
    exe = shutil.which(binary)
    if not exe:
        raise RuntimeError(
            "LeanCTX CLI is not installed or not on PATH. Install yvgude/lean-ctx first."
        )

    payload = {"command": command, "output": raw_output}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        payload_path = Path(handle.name)
    try:
        proc = subprocess.run(
            [
                exe,
                "call",
                "ctx_compare",
                "--project-root",
                str(Path(project_root).resolve()),
                "--json-file",
                str(payload_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    finally:
        payload_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"LeanCTX ctx_compare failed:\n{proc.stdout[-10000:]}")
    report = proc.stdout or ""
    if "compress preview" not in report:
        raise RuntimeError(f"Unexpected LeanCTX ctx_compare output:\n{report[-8000:]}")

    compressed = _reconstruct_from_preview(raw_output, report)
    original_tokens, compressed_tokens, saved_pct = _parse_token_header(report)
    return LeanCtxResult(
        text=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        saved_percent=saved_pct,
        original_bytes=len(raw_output.encode()),
        compressed_bytes=len(compressed.encode()),
        report=report,
    )


def _parse_token_header(report: str) -> tuple[int | None, int | None, float | None]:
    m = re.search(
        r"tokens:\s*(\d+)\s*->\s*(\d+)\s*\(-\d+,\s*([0-9.]+)%\s+saved\)",
        report,
    )
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def _reconstruct_from_preview(original: str, report: str) -> str:
    """Reconstruct the exact compressed line sequence from LeanCTX's line diff."""
    marker = "-- diff (original -> compressed) --"
    if marker not in report:
        raise RuntimeError("LeanCTX preview is missing its diff section")
    diff = report.split(marker, 1)[1].strip("\n")
    if diff.strip() == "(no changes)":
        return original

    deletes: set[int] = set()
    additions: dict[int, list[str]] = {}
    for line in diff.splitlines():
        if line.startswith("diff +"):
            break
        m = re.match(r"^([+-])(\d+):(?:\s?)(.*)$", line)
        if not m:
            continue
        sign, index_s, text = m.groups()
        index = int(index_s)
        if sign == "-":
            deletes.add(index)
        else:
            additions.setdefault(index, []).append(text)

    old_lines = original.splitlines()
    kept = [line for idx, line in enumerate(old_lines, 1) if idx not in deletes]
    target_len = len(kept) + sum(len(v) for v in additions.values())
    target: list[str | None] = [None] * target_len

    for one_based, values in sorted(additions.items()):
        start = max(0, one_based - 1)
        for offset, value in enumerate(values):
            pos = start + offset
            if pos >= len(target):
                target.extend([None] * (pos - len(target) + 1))
            if target[pos] is None:
                target[pos] = value
            else:
                target.insert(pos, value)

    kept_iter = iter(kept)
    for i, value in enumerate(target):
        if value is None:
            try:
                target[i] = next(kept_iter)
            except StopIteration:
                target[i] = ""
    target.extend(list(kept_iter))
    return "\n".join(str(v) for v in target)
