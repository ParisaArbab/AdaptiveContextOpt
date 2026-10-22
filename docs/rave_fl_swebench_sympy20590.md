# RAVE-FL on a real SWE-bench Python bug

First instance: `sympy__sympy-20590` (SWE-bench Lite). The Java-only `rave_fl.pilot` **does not** support this bug; use `rave_fl.swebench_pilot`.

## 1. Two evidence conditions, not interchangeable

**Cloud CPU smoke test:** GitHub Actions workflow `.github/workflows/rave-swebench-one-bug.yml` downloads the official SWE-bench Lite metadata, checks out the exact `base_commit` of SymPy, prepares real Python AST function sketches, then runs a local small Ollama model `qwen2.5:0.5b`. Uses **`--evidence issue_only`**: the official issue and failing test *names*, **no raw observed test log**. That is not the same setup as our earlier debugging-log experiments. It is not an official SWE-bench patch-resolution evaluation.

**AIScalar research test:** Prefer `--evidence raw` on the original benchmark's saved `outputs/raw_test_output.txt`. This is an observed test-log condition. Never create an invented failure log to satisfy the runner. Use the same evidence file for every policy in one paired run.

## 2. AIScalar commands

```bash
ssh parbab@aiscalar.cdm.depaul.edu
cd ~/AdaptiveContextOpt
git fetch origin
git switch codex/rave-fl-adaptive-pilot
git pull --ff-only origin codex/rave-fl-adaptive-pilot
python -m pip install -r requirements-rave-pilot.txt

WORK=data/swebench_workspaces/sympy__sympy-20590
test -f "$WORK/metadata.json" || { echo "Missing SWE-bench metadata"; exit 1; }
test -d "$WORK/repo" || { echo "Missing buggy checkout"; exit 1; }
test -s "$WORK/outputs/raw_test_output.txt" || { echo "Missing real raw test output"; exit 1; }

RUN="results/swebench_sympy20590_preflight_$(date -u +%Y%m%dT%H%M%SZ)"
python -m rave_fl.swebench_pilot --work "$WORK" --evidence raw \
  --source-package sympy --candidate-source lexical --top-k 12 \
  --arms full,fixed_batch,repair_utility,repair_grounded \
  --dry-run --output "$RUN"
```

This first command should produce `run.json` and `input.json` without making any model calls or reading the gold patch. Review its candidate list before spending compute. The current baseline is **lexical Python function retrieval**, not SieveFL, Graphify or semantic-embedding retrieval.

If the earlier SWE-bench pipeline generated `$WORK/outputs/agent4lr/candidate_pools.json`, use `--candidate-source existing_pool --pool-condition raw` instead to evaluate its **existing candidate pool** (first occurrence is kept for duplicates). The runner fails on unresolvable source entities rather than silently guessing.

## 3. Live LLM call (real raw-log condition)

First verify the model and Ollama service exist on your selected server port:

```bash
curl -fsS http://127.0.0.1:11436/api/tags
```

Then, **only if the tag `qwen2.5:7b` is installed on that Ollama server**, run:

```bash
RUN="results/swebench_sympy20590_raw_qwen7b_$(date -u +%Y%m%dT%H%M%SZ)"
python -m rave_fl.swebench_pilot \
  --work data/swebench_workspaces/sympy__sympy-20590 \
  --evidence raw --source-package sympy --candidate-source lexical --top-k 12 \
  --model qwen2.5:7b --base-url http://127.0.0.1:11436 \
  --arms full,fixed_batch,repair_utility,repair_grounded \
  --full-budget 60000 --policy-budget 16000 \
  --max-full-repairs 4 --batch-size 4 --max-output-tokens 160 \
  --context-window 32768 --timeout 1800 \
  --output "$RUN"
```

Each arm saves **actual** Ollama usage counts, response ledgers, state and ranked candidates. `run.json` is the experiment summary. A nonempty output folder is refused to prevent accidentally overwriting results.

## 4. Interpretation

- The verifier uses **Python** prompts and AST sketches, with L0 signatures, L1 lossy structural summaries and complete L2 source.
- The initial candidate retrieval is **not** equivalent to SieveFL; if the true faulty function is absent from the retrieved pool, verification cannot recover it.
- Full verifies every candidate only **within its explicit budget**. A budget-limited Full run is not a complete accuracy reference.
- The existing risk predictor and evidence-grounded rescue are **untrained**. Do not claim an accuracy improvement before examining real results.
- The offline evaluation reads metadata `patch` **only after all model arms finish**. It maps patch hunk positions back to the checked-out Python function index, which is a **weak function-level proxy**, not official SWE-bench patch resolution. Class/module-level changes may have no function gold label at all.
- The cloud trial uses a smaller model and issue-only evidence, so compare it only internally against arms from the **same cloud trial**, not against the Qwen 7B raw-log run.
- One inspected SWE-bench case is a **development pilot**, not a held-out generalization result.

## 5. GitHub evidence

The [one-bug workflow](https://github.com/ParisaArbab/AdaptiveContextOpt/actions/workflows/rave-swebench-one-bug.yml) records the official instance/commit, a gold-free preflight, real provider calls and an artifact with the complete experiment ledgers when completed. The [unit-test workflow](https://github.com/ParisaArbab/AdaptiveContextOpt/actions/workflows/rave-fl-pilot.yml) tests Python extraction, syntax failures, evaluator prompt language and no-inference preflight behavior.
