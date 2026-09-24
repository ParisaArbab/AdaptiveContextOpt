# AdaptiveContextOpt

Smart context optimization for AI fault localization.

The `main` branch now runs the Defects4J workflow shown in the project architecture:

**Defects4J -> Graphify -> RAW vs LeanCTX -> Agent4SR -> SBIR/Ochiai/BoostN merge -> Agent4LR -> Top-5 evaluation**

The main research question is simple:

> How aggressively can LeanCTX compress debugging context while a feedback loop selectively recovers only the localization-critical evidence needed to preserve fault localization?

## 1. Reference repositories

Clone the three codebases used to build and verify this implementation:

```bash
./scripts/clone_reference_repos.sh
```

This creates:

```text
references/
  graphify/
  FlexFL_OriginalReplication/
  lean-ctx/
```

They are ignored by Git because they are external references, not copied source code.

## 2. Required tools

The machine running the benchmark needs:

- Python 3.10+
- Defects4J, with `defects4j` on `PATH`
- Java required by the chosen Defects4J project
- Graphify CLI, installed from package `graphifyy`
- LeanCTX CLI, installed from the upstream `yvgude/lean-ctx` project
- an LLM backend, Ollama is the default

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Verify the external commands:

```bash
defects4j info -p Time
graphify --help
lean-ctx --version
ollama list
```

## 3. Models

The architecture uses three model families:

```text
Llama 3
Qwen 2
Mistral
```

Default Ollama tags in the runner are:

```text
llama3:8b,qwen2:7b,mistral:7b
```

If your installed model tags are different, pass the exact names shown by `ollama list`.

## 4. First run, one bug

Start with `Time-25`:

```bash
./scripts/run_defects4j_benchmark.sh
```

Or explicitly:

```bash
python3 wp1/run_wp1_benchmark.py \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --backend ollama \
  --models 'llama3:8b,qwen2:7b,mistral:7b' \
  --bugs Time-25
```

To test a single model first:

```bash
python3 wp1/run_wp1_benchmark.py \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --backend ollama \
  --models 'llama3:8b' \
  --bugs Time-25
```

## 5. Multiple bugs

```bash
python3 wp1/run_wp1_benchmark.py \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --models 'llama3:8b,qwen2:7b,mistral:7b' \
  --bugs 'Time-25,Chart-1,Lang-1' \
  --continue-on-error
```

To run every bug that has all three traditional FL result files in the FlexFL replication:

```bash
python3 wp1/run_wp1_benchmark.py \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --all-flexfl-bugs \
  --models 'llama3:8b,qwen2:7b,mistral:7b' \
  --continue-on-error
```

## 6. Controlled RAW vs LeanCTX comparison

For each bug, the runner does this only once:

```bash
defects4j test
```

It saves the complete stdout/stderr as `raw_output.txt`.

Then LeanCTX receives that same captured text through its real `ctx_compare` shell pipeline. Therefore both arms start from exactly the same test output.

There is no fake Python compressor and no silent fallback.

## 7. FlexFL candidate flow

For each condition and model:

```text
Agent4SR -> top 5
SBIR     -> top 5
Ochiai   -> top 5
BoostN   -> top 5
                 |
                 v
          merge up to 20
                 |
                 v
             Agent4LR
                 |
                 v
           final top 5
```

The merge order matches the original replication's `combine.py`:

```text
SBIR -> Ochiai -> BoostN -> Agent4SR
```

## 8. Results

Each run is stored under:

```text
results/<UTC-run-id>/
```

Important files include:

```text
config.json
per_run.csv
summary.json
<bug>/raw_output.txt
<bug>/leanctx_output.txt
<bug>/leanctx_preview.txt
<bug>/compression.json
<bug>/graphify_structure.json
<bug>/<model>/raw/agent4sr.json
<bug>/<model>/raw/merged_candidates.json
<bug>/<model>/raw/agent4lr.json
<bug>/<model>/raw/evaluation.json
<bug>/<model>/leanctx/...
```

`summary.json` reports Top-1, Top-3, Top-5, MAP, and MRR by model and condition. It also lists **compression-tax** cases where RAW finds a faulty method in the final Top-5 but LeanCTX does not.

## 9. Adaptive feedback research mode

For SWE-bench density experiments, the main branch includes a gold-free,
evidence-guided adaptive feedback loop. It starts from a compressed LeanCTX
context, runs Agent4SR with Graphify, and asks a separate evaluator to name the
specific unresolved evidence needed to support the localization.

The controller now uses a feedback-guided targeted-recovery policy:

```text
compressed context
  -> Agent4SR
  -> feedback identifies one concrete missing evidence need
  -> retrieve only that runtime/source/structure evidence
  -> rerun Agent4SR at the SAME LeanCTX density
  -> repeat only while new targeted evidence is available

if no concrete new evidence can be named or retrieved:
  -> stop without restoring more of the original context
```

The default research policy is therefore **not** to reconstruct RAW context and
not to increase LeanCTX density. The feedback loop is designed to recover only
small, localization-critical evidence units that compression may have removed.

Targeted evidence can include a local runtime failure excerpt, a concrete
`file.py::Entity` source snippet, a bounded transitive inheritance chain with
explicit `__slots__` status, or the production entities in a known source
file. Requests that mention parents, base classes, ancestors, MRO, or
inheritance are normalized to `inheritance_chain` even if the evaluator labels
them as a plain source snippet. Retrieved evidence is stored in an evidence
ledger and is not requested again.

Each retry also receives the previous Agent4SR Top-5 as hypothesis memory. The
ranking is not treated as ground truth, but it is carried forward so a new
search trajectory does not silently discard useful localization state without
new contradictory evidence.

The default `--recovery-policy targeted_only` keeps the initial LeanCTX density
fixed for the complete feedback loop. If the evaluator cannot name a concrete
missing evidence target, or retrieval produces no new evidence, the run stops
instead of restoring more context. The older density-fallback behavior remains
available only for comparison with
`--recovery-policy targeted_then_density`.

`--max-feedback-rounds 3` means one initial Agent4SR run plus at most three
feedback-driven targeted-recovery retries. The feedback evaluator and retrieval
controller never receive the gold faulty entity or patch. Gold remains
evaluation-only.

This mode uses the research density helper compiled from the pinned LeanCTX
source tree. It is intentionally labeled research mode because it calls
LeanCTX's density algorithm directly rather than the production
`ctx_compare` shell safety path.

Example:

```bash
python -m wp1.run_swebench_agent4sr_feedback \
  --instance sympy__sympy-20590 \
  --model qwen3.6:27b \
  --initial-density 0.30 \
  --density-schedule 0.30 \
  --recovery-policy targeted_only \
  --max-feedback-rounds 3
```

Results are written under
`data/swebench_workspaces/<instance>/outputs/adaptive_feedback/`.

## 10. Main modules

| File | Purpose |
|---|---|
| `wp1/defects4j_harness.py` | checkout bugs and capture test output once |
| `wp1/graphify_structure.py` | build/query the Graphify source graph |
| `wp1/leanctx_compressor.py` | call the real LeanCTX production shell compressor |
| `wp1/flexfl_data.py` | load FlexFL traditional FL results, inputs, and ground truth |
| `wp1/flexfl_pipeline.py` | run Agent4SR, merge candidates, run Agent4LR |
| `wp1/evaluation.py` | Top-k, MAP, MRR |
| `wp1/llm_backends.py` | Ollama, OpenAI-compatible, and Anthropic access |
| `wp1/run_wp1_benchmark.py` | end-to-end benchmark runner |
| `wp1/adaptive_feedback.py` | gold-free feedback evaluator and selective targeted evidence recovery |
| `wp1/leanctx_density.py` | wrapper for the pinned LeanCTX density research helper |
| `wp1/run_swebench_agent4sr_feedback.py` | targeted-evidence-first adaptive Agent4SR loop |

More details are in `docs/architecture.md` and `docs/reference_alignment.md`.
