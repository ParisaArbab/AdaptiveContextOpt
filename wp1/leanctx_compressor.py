"""
leanctx_compressor.py — WP1

Replaces rtk_compressor.py as the "Static/Smart" compression condition.

IMPORTANT — architecture note (verified against the real yvgude/lean-ctx repo
and the lean-ctx-sdk 0.3.0 source): lean-ctx is NOT a pure text-in/text-out
library. It is a local Rust daemon (`lean-ctx serve` / `lean-ctx proxy
enable`) that exposes an HTTP API; `lean-ctx-sdk`'s `ProxyClient.compress()`
is a thin client over that daemon's `/v1/compress` endpoint. There is no
offline "just call a function" path in the shipped SDK — confirmed by
reading ProxyClient._send(), which raises LeanCtxConnectionError telling you
to run `lean-ctx proxy enable` if nothing answers at base_url.

This mirrors how rtk was handled: rtk_compressor.py is a Python
re-implementation of rtk's *documented* heuristics, sourced from its README
and a GitHub issue, used because running the real Rust binary inside the
benchmark loop wasn't the point — reproducing its documented behavior
faithfully was. We do the same thing here, with the same caveat stated
explicitly in every result this module produces:

  MODE "daemon"    — real lean-ctx binary running locally, called via the
                     official lean-ctx-sdk. This is the authoritative mode
                     and the only one whose numbers should go in the formal
                     deliverable. Requires `lean-ctx serve` (or
                     `lean-ctx proxy enable`) to be running — install via
                     https://leanctx.com/install.sh, `cargo install lean-ctx`,
                     or `npm install -g lean-ctx-bin`.
  MODE "reference" — Python re-implementation of lean-ctx's documented
                     density-mode algorithm ("target density: SDE-style
                     budget compression — keeps the highest-entropy lines
                     until ~X% of original tokens remain, deterministic",
                     per the lean-ctx README's Compression section). Used
                     ONLY to validate the rest of the pipeline (agent
                     interface, feedback loop, scoring) before the daemon is
                     reachable. Every output is tagged mode="reference" so
                     compression_tax_analyzer.py can exclude it from final
                     numbers, exactly like the rtk baseline required a real
                     Docker capture before its numbers were trusted.

Attempted in this environment: the official installer, `npm install -g
lean-ctx-bin`, and the GitHub releases API all failed here because of a
GitHub API rate limit on the sandbox's shared IP (confirmed: 403 on
api.github.com/repos/yvgude/lean-ctx/releases/latest, "API rate limit
exceeded"). That's a sandbox network limitation, not a lean-ctx problem —
running the installer on your own machine or CI runner (unique IP) should
succeed normally.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

try:
    from lean_ctx import ProxyClient, LeanCtxConnectionError
except ImportError:  # pragma: no cover
    ProxyClient = None
    LeanCtxConnectionError = Exception


@dataclass
class CompressionResult:
    text: str
    mode: str  # "daemon" | "reference"
    original_tokens_est: int
    compressed_tokens_est: int
    reduction_pct: float


def _estimate_tokens(text: str) -> int:
    # rough, consistent estimator (chars/4) — same heuristic used throughout
    # WP1 so raw/rtk/lean-ctx reduction percentages are comparable to each other
    return max(1, len(text) // 4)


def _daemon_compress(text: str, model: Optional[str] = None) -> Optional[str]:
    """Try the real local lean-ctx daemon. Returns None if unreachable."""
    if ProxyClient is None:
        return None
    try:
        client = ProxyClient(timeout=2.0)
        messages = [{"role": "tool", "content": text}]
        result = client.compress(messages, model=model)
        return result.messages[0]["content"] if result.messages else text
    except LeanCtxConnectionError:
        return None
    except Exception:
        return None


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
    model: Optional[str] = None,
    force_reference: bool = False,
) -> CompressionResult:
    """Main entry point used by run_wp1_benchmark.py for the lean-ctx condition."""
    original_tokens = _estimate_tokens(text)

    compressed_text = None
    mode = "reference"
    if not force_reference:
        compressed_text = _daemon_compress(text, model=model)
        if compressed_text is not None:
            mode = "daemon"

    if compressed_text is None:
        compressed_text = _reference_density_compress(text, target_density=target_density)
        mode = "reference"

    compressed_tokens = _estimate_tokens(compressed_text)
    reduction = 100.0 * (1 - compressed_tokens / original_tokens) if original_tokens else 0.0

    return CompressionResult(
        text=compressed_text,
        mode=mode,
        original_tokens_est=original_tokens,
        compressed_tokens_est=compressed_tokens,
        reduction_pct=round(reduction, 1),
    )


if __name__ == "__main__":
    import sys

    if "--stdin" in sys.argv:
        sample = sys.stdin.read()
    else:
        sample = (
            "===== FAILURES =====\n"
            "test_separable.py::test_coord_matrix FAILED\n"
            "    assert result.tolist() == expected.tolist()\n"
            "AssertionError: arrays differ at index [2][1]: True != False\n"
            "-----------------------------\n"
        )
    r = compress(sample)
    print(f"[mode={r.mode}] {r.original_tokens_est} -> {r.compressed_tokens_est} tok "
          f"({r.reduction_pct}% reduction)\n")
    print(r.text)
