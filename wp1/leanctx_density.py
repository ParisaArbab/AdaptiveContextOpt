"""Research wrapper for LeanCTX's entropy density helper.

This module does not reimplement LeanCTX compression. It invokes an external
helper built from the pinned LeanCTX source tree and records the actual token
counts reported by that helper. This is research mode, not production
``ctx_compare`` shell compression.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import subprocess
import tempfile


@dataclass
class DensityCompressionResult:
    text: str
    target_density: float
    original_tokens: int
    compressed_tokens: int
    saved_percent: float
    helper: str
    report: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("text", None)
        return data


def resolve_density_helper(path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.getenv("LEANCTX_DENSITY_HELPER")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(
        Path.home()
        / "lean-ctx-research/rust/target/release/examples/density_research"
    )

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        "LeanCTX density helper was not found or is not executable. Tried:\n"
        + tried
        + "\nPass --density-helper or set LEANCTX_DENSITY_HELPER."
    )


def compress_to_density(
    raw_output: str,
    density: float,
    *,
    helper: str | Path | None = None,
    timeout: int = 180,
) -> DensityCompressionResult:
    if not 0.0 < density <= 1.0:
        raise ValueError("density must satisfy 0 < density <= 1")

    helper_path = resolve_density_helper(helper)
    with tempfile.TemporaryDirectory(prefix="leanctx-density-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "raw.txt"
        output_path = tmp_path / "compressed.txt"
        input_path.write_text(raw_output)

        proc = subprocess.run(
            [str(helper_path), str(input_path), str(density), str(output_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "LeanCTX density helper failed:\n" + (proc.stdout or "")[-8000:]
            )
        if not output_path.exists():
            raise RuntimeError("LeanCTX density helper produced no output file")
        text = output_path.read_text(errors="replace")

    match = re.search(
        r"original=(\d+)\s+compressed=(\d+)\s+saved=([0-9.]+)%",
        proc.stdout or "",
    )
    if not match:
        raise RuntimeError(
            "Could not parse LeanCTX density helper metrics:\n"
            + (proc.stdout or "")[-4000:]
        )

    original_tokens = int(match.group(1))
    compressed_tokens = int(match.group(2))
    saved_percent = float(match.group(3))
    return DensityCompressionResult(
        text=text,
        target_density=float(density),
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        saved_percent=saved_percent,
        helper=str(helper_path),
        report=proc.stdout or "",
    )
