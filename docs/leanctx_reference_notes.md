# lean-ctx reference notes

Source of truth: [yvgude/lean-ctx](https://github.com/yvgude/lean-ctx) (README,
`lean_ctx/client.py`, `lean_ctx/discovery.py` — read directly from the
installed `lean-ctx-sdk==0.3.0` package on 2026-08-25).

## What it actually is

lean-ctx is **not** a text-in/text-out compression function. It's a local
Rust daemon (`lean-ctx serve` / `lean-ctx proxy enable`) that runs
persistently and exposes an HTTP API (default `http://127.0.0.1:<uid-derived
port>`, base 4444). `lean-ctx-sdk`'s `ProxyClient` is a thin client over that
daemon's `/v1/compress` endpoint — confirmed by reading `ProxyClient._send()`,
which raises `LeanCtxConnectionError` with the message "Is the daemon
running? Try: lean-ctx proxy enable" when nothing answers.

This is architecturally different from rtk, which our original
`rtk_compressor.py` could faithfully reimplement as a pure function because
rtk's documented behavior is a stateless text transform. lean-ctx's actual
compression logic lives inside the Rust binary and is not shipped as an
invokable Python algorithm.

## Two-mode design in `leanctx_compressor.py`

- **`daemon` mode** (authoritative): calls the real local daemon via
  `ProxyClient().compress()`. Only this mode's numbers belong in the formal
  Compression Tax report.
- **`reference` mode** (pipeline validation only): a Python re-implementation
  of the *documented* density-mode contract — "keeps the highest-entropy
  lines until ~X% of original tokens remain, deterministic" (README,
  Compression section) — using Shannon entropy per line as the retention
  score. Every result is tagged `mode="reference"` and
  `compression_tax_analyzer.py` marks it `provisional=True`, excluding it
  from headline numbers automatically.

## Why daemon mode isn't live yet in this sandbox

Three install paths were attempted on 2026-08-25 and all failed for the same
underlying reason:

| Method | Result |
|---|---|
| `curl -fsSL https://leanctx.com/install.sh \| sh` | `curl: (22) ... 403` on the GitHub releases API |
| `npm install -g lean-ctx-bin` | preinstall script hit the same GitHub releases fetch, timed out |
| `curl api.github.com/repos/yvgude/lean-ctx/releases/latest` (direct) | `403: API rate limit exceeded for 34.23.141.224` |

This is a GitHub API rate limit on the sandbox's shared/ephemeral IP, not a
lean-ctx problem. Running the same installer from a normal machine or CI
runner (its own IP, or an authenticated `GITHUB_TOKEN` for a higher rate
limit) should succeed normally. Once installed, `daemon` mode activates
automatically — `leanctx_compressor.py` requires zero code changes, since it
already tries `ProxyClient()` first and only falls back to `reference` mode
on `LeanCtxConnectionError`.

## Action item before final WP1 numbers

Run, on a machine with normal GitHub API access:

```bash
curl -fsSL https://leanctx.com/install.sh | sh
lean-ctx proxy enable
lean-ctx doctor   # confirm the daemon answers
```

Then re-run `run_wp1_benchmark.py` — every `leanctx` condition outcome
should switch from `compressor_mode: "reference"` to `"daemon"`, and only
then are the numbers publishable.
