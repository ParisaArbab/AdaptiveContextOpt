# SWE-bench Lite diagnostic run

Run from the project root. The input contains the original
`sympy__sympy-17139` instance from the supplied ZIP, including its base commit
and test patch. No dataset fetch is needed.

Install the pinned official compressor once:

```bash
python3 scripts/install_leanctx.py
```

The installer verifies the release SHA256 and installs under
`.tools/leanctx/3.10.1/`, without changing editor settings or installing a
global proxy. The adapter automatically uses that binary; `LEAN_CTX_BINARY`
can explicitly select another installation. Full now fails if real LeanCTX
fails; it does not silently substitute the reference compressor.

Compression uses upstream `ctx_compare`'s production shell engine with the
captured test command. Its complete diff, byte counts, executable path and
executable SHA256 are saved in `compression.json`. Reconstruction must match
upstream's reported output bytes. `--target-density` only applies to explicit
reference-mode callers; upstream shell compression is command-aware.

```bash
bash scripts/run_pipeline.sh \
  --llm ollama --model llama3.1:8b \
  --dataset swe-bench-lite \
  --instances data/smoke_sympy_17139.json \
  --arms full,pure_flexfl --n 1 --limit 1 \
  --local-fallback --preflight
```

The runner uses `data/repos/sympy__sympy-17139` as its managed checkout;
it prepares the checkout and its dependencies if missing. The dataset JSON
and the executable checkout are separate inputs. Existing checkout reuse
follows the harness's existing reset/clean behavior.

Each run creates a timestamped directory under `results/`. Send that entire
directory, including `traces/`, for review. `run.log` alone is insufficient.

Saved artifacts:

- `run.log`: unbuffered console stdout/stderr from the shell pipeline.
- `traces/config.json`, `instances.json`, `git_commit.txt`, `source/`:
  invocation, frozen input and the actual Python/shell sources, including
  uncommitted source changes.
- `traces/run.jsonl`: progress and capture/graph/arm failures.
- `traces/<instance>/capture.json`, `raw_test_output.txt`: the test command,
  exit status, complete captured output, notes and errors. Coverage is saved
  once as `coverage.json`, referenced by `coverage_artifact` in the capture.
- `traces/<instance>/wp1coverage-run.json`: complete stdout/stderr and command
  from the coverage run when coverage was attempted.
- `traces/<instance>/structure_map.json`, `call_graph.json`: the structural
  inputs shared by the arms.
- `traces/<instance>/<arm>/events.jsonl`: full logical LLM request prompts,
  responses, context texts, stage, timestamps, latency and errors. Provider
  response metadata includes finish reason and usage when supplied. Tool
  results are visible in subsequent prompts and in `localization.json`.
- `compression.json`, `localization.json`, `outcome.json` in each arm:
  compression details, protocol/tool history, resolved candidates, merge
  provenance, scores and token report. An arm error is saved in `error.txt`.

Events are flushed as they occur; outcomes are checkpointed after each arm.
If interrupted, the last request and completed arms remain reviewable.
The logging does not alter prompts, the merge order or stage ordering.

The headline LLM token report uses provider usage when present, with
`llm_token_source` identifying the basis. Context-size buckets still use the
client tokenizer; tiktoken is not an exact Llama tokenizer. Provider usage is
also saved per response. Native Ollama requests explicitly set `num_ctx` to
16384 by default; pass `--context-window` to change it. Both arms use the same
capacity. A response reporting a prompt at that limit is rejected as possible
truncation and enters the existing adaptive-MAX path.
SDK-internal HTTP retry attempts are not individually exposed by this trace.

Review should verify that the intended failing tests actually execute, Ochiai
has failing and passing contexts, Agent4SR produces resolvable candidates,
Agent4LR sees the evidence and ranks the merged candidates, and structural
briefing size stays bounded. Passing the logging tests or getting a nonempty
ranking alone does not demonstrate correct localization or token savings.
