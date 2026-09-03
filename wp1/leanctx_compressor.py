"""LeanCTX compression of a captured test output.

Two modes, and which one ran is recorded on every result:

  "cli"       — the real LeanCTX binary, driven through `lean-ctx call
                ctx_compare` (merged from the Defects4J branch, which got
                this working first). Authoritative: only these numbers
                belong in the formal Compression Tax report.
  "reference" — a Python re-implementation of LeanCTX's *documented*
                density-mode contract ("keeps the highest-entropy lines
                until ~X% of the original tokens remain, deterministic").
                Tagged provisional everywhere downstream and excluded from
                headline numbers by the analyzer.

The Defects4J branch deliberately had no fallback: for a single-arm run,
failing loudly beats silently changing the experiment. That reasoning does
not carry over to the ablation study, where the compressor is one of three
variables under test across five or six arms — there, an uninstalled binary
would take out every leanctx arm and leave nothing to compare the other arms
against. So the fallback exists, and the cost of using it is paid in
labelling rather than in silence: `mode="reference"` propagates to
`provisional=True` on the outcome, the analyzer excludes those arms from its
headline table, and the plots drop a PROVISIONAL.txt naming the reason.

Use `compress()` for the ablation pipeline; `compress_captured_shell_output()`
is the lower-level real-binary call and raises rather than falling back.
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


# ---------------------------------------------------------------------------
# Reference mode + the ablation pipeline's entry point
# ---------------------------------------------------------------------------

import math
from collections import Counter
from typing import Optional


@dataclass
class CompressionResult:
    text: str
    mode: str                    # "cli" | "reference"
    original_tokens_est: int
    compressed_tokens_est: int
    reduction_pct: float
    detail: dict | None = None


def _estimate_tokens(text: str) -> int:
    """Byte-level estimate used only inside this module, so a compression
    ratio can be reported even when LeanCTX itself reports no token header.
    The study's real token numbers come from token_meter's tokenizer, not
    from here."""
    return max(1, len(text) // 4)


def _line_entropy(line: str) -> float:
    """Shannon entropy over characters — proxy for 'information density' per
    lean-ctx's documented density-mode description (highest-entropy lines
    kept first)."""
    if not line.strip():
        return 0.0
    counts = Counter(line)
    n = len(line)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# Lines that are near-always noise regardless of entropy score: separators,
# progress bars, timing footers. Kept as a small, documented allowlist rather
# than a broad regex so it doesn't accidentally eat real assertion output.
_NOISE_PATTERNS = [
    re.compile(r"^=+\s.*\s=+$"),          # pytest "===== ... ====" banners
    re.compile(r"^-+$"),                   # bare separator lines
    re.compile(r"^\s*$"),
]


def _reference_density_compress(text: str, target_density: float = 0.4) -> str:
    """Documented-behavior re-implementation of lean-ctx's density mode.

    Per lean-ctx README: 'Target density (density:0.4): SDE-style budget
    compression — keeps the highest-entropy lines until ~40% of the original
    tokens remain, deterministic.' We reproduce that contract: score every
    line, keep the highest-entropy ones (skipping pure-noise banners) until
    the running token estimate hits target_density * original, preserving
    original order.
    """
    lines = text.splitlines()
    original_tokens = _estimate_tokens(text)
    budget = int(original_tokens * target_density)

    scored = []
    for i, line in enumerate(lines):
        if any(p.match(line) for p in _NOISE_PATTERNS):
            score = -1.0  # always dropped first
        else:
            score = _line_entropy(line)
        scored.append((score, i, line))

    scored.sort(key=lambda t: t[0], reverse=True)

    kept_idx = set()
    used = 0
    for score, i, line in scored:
        cost = _estimate_tokens(line)
        if used + cost > budget and kept_idx:
            continue
        kept_idx.add(i)
        used += cost
        if used >= budget:
            break

    out_lines = []
    last_kept = -2
    for i, line in enumerate(lines):
        if i in kept_idx:
            if i != last_kept + 1 and out_lines:
                out_lines.append("... [lean-ctx reference-mode: lines omitted, density budget] ...")
            out_lines.append(line)
            last_kept = i

    return "\n".join(out_lines)


def compress(
    text: str,
    target_density: float = 0.4,
    project_root: Optional[Path] = None,
    command: str = "pytest",
    force_reference: bool = False,
) -> CompressionResult:
    """Entry point for run_wp1_benchmark's leanctx arms.

    Tries the real LeanCTX binary first and falls back to reference mode,
    recording which one ran. A caller that needs the real thing or nothing
    should call compress_captured_shell_output() directly.
    """
    if not force_reference:
        try:
            real = compress_captured_shell_output(
                text, project_root or Path("."), command=command
            )
            original = real.original_tokens or _estimate_tokens(text)
            compressed = real.compressed_tokens or _estimate_tokens(real.text)
            reduction = (real.saved_percent
                         if real.saved_percent is not None
                         else 100.0 * (1 - compressed / original) if original else 0.0)
            return CompressionResult(
                text=real.text, mode="cli",
                original_tokens_est=original, compressed_tokens_est=compressed,
                reduction_pct=round(reduction, 1), detail=real.to_dict(),
            )
        except Exception:
            # Binary missing, or ctx_compare rejected this input. Either way
            # the run continues in reference mode, tagged as such.
            pass

    compressed_text = _reference_density_compress(text, target_density=target_density)
    original = _estimate_tokens(text)
    compressed = _estimate_tokens(compressed_text)
    return CompressionResult(
        text=compressed_text, mode="reference",
        original_tokens_est=original, compressed_tokens_est=compressed,
        reduction_pct=round(100.0 * (1 - compressed / original) if original else 0.0, 1),
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
