# Real LeanCTX integration and evaluation — 2026-09-10

The official [LeanCTX v3.10.1 release](https://github.com/yvgude/lean-ctx/releases/tag/v3.10.1)
is installed under `.tools/leanctx/3.10.1/`. Its release archive SHA256 was
verified. The adapter invokes `ctx_compare`, which calls upstream's production
shell compressor. No global proxy or editor integration was installed.

The pipeline remains Graphify → LeanCTX → compression-adjustment feedback →
FlexFL localization / GraphLocator → evaluation. RAW is `pure_flexfl`.
Real compression is mandatory by default; errors no longer trigger automatic
reference-mode substitution. The actual harness command and repository root
are passed to the compressor by the benchmark runner. The real shell engine
does not expose the reference implementation's target-density control.

## Measured result

One instance: `sympy__sympy-17139`; local model: `llama3.1:8b`;
context capacity: 16,384; output limit: 1,024. Both arms were freshly run.

| Measure | Full, real LeanCTX | RAW |
| --- | ---: | ---: |
| Actual provider LLM tokens | 87,361 | 52,615 |
| Model calls | 21 | 15 |
| Method Top-1 / Top-3 / Top-5 | 1 / 1 / 1 | 1 / 1 / 1 |
| MRR | 1 | 1 |
| Agent4SR resolved candidates | 5 | 5 |

Both rank `sympy/simplify/fu.py::_TR56._f()` first. Ochiai uses two failing
and 40 passing test contexts. On this instance, end-to-end token savings are
**−66.0%**: Full consumes 34,746 more tokens, not fewer.

## Where compression savings went

These input-size counts use LeanCTX/client tokenization, not the exact Llama
provider tokenizer:

- Raw capture: 1,629 tokens.
- After real LeanCTX: 203 tokens (**87.5% initial reduction**).
- After feedback: 1,621 tokens (**0.49% net input reduction** before briefing).
- Structural briefing: 443 additional tokens.

The compressor removed failure evidence that feedback restored. Total LLM
usage includes every growing search transcript, not just this initial input:

| Stage | Full tokens | RAW tokens |
| --- | ---: | ---: |
| Feedback | 2,904 | 0 |
| Agent4SR | 59,263 | 31,883 |
| Agent4LR | 25,194 | 20,732 |

Graph processing ran but contributed no additional model calls in this run.
Most of the excess comes from the longer Agent4SR search. Thus real LeanCTX
is integrated and localization succeeds here, while end-to-end optimization
is not demonstrated. This result must not be presented as token savings.

## Evidence and scope

Artifacts: `results/verification_20260910_real_leanctx/`, including `run.log`,
`verify.py`, frozen sources, shared graph/index, coverage, raw capture, per-arm
full request/response events, compression preview, executable hash, outcomes,
and `artifact_validation.json`.

This is a controlled replay of the supplied failure capture, with coverage
recollected after its context fix. Graphify and both localization arms run
on the buggy checkout with only the test patch applied. The replay's shell
compression command is `python bin/test --no-subprocess`. It is not a fresh
official Docker-harness run or a complete SWE-bench Lite evaluation.

All 24 tests passed, including a real-binary integration test and rejection
of missing LeanCTX rather than fallback. Saved provider usage sums match
reported totals; all model requests have responses, no transport errors were
recorded, and Agent4LR's first five predictions stay inside the fixed merge.

To install and run the normal capture-to-evaluation path again, follow
[smoke_run_logging.md](smoke_run_logging.md). More instances are required to
estimate benchmark-wide accuracy and token savings.
