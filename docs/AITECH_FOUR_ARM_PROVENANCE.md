# AITECH Experiment 3: Four-arm source provenance ablation

## Purpose

The first AIScalar run with `qwen3.6:27b` for the real-source
`sympy__sympy-20590` microcase produced:

| Arm | File top-1 | Entity top-1 | Ollama input tokens |
|---|---|---|---:|
| A: Raw seven-line excerpt | correct | correct | 3,563 |
| B: Relevance-only | incorrect | incorrect | 2,683 |
| C: Dependency-aware | incorrect | correct (`Printable`) | 2,553 |

C predicted `sympy/printing/printer.py::Printable`, rather than
`sympy/core/_print_helpers.py::Printable`. These are user-supplied
**single-run preliminary observations**; they are not a general benchmark.

**New D arm:** The same selected evidence IDs as C with verified
`[source: file.py:line]` annotations taken from archived
`get_code_snippet` outputs. The provenance extractor does not use the
gold file, gold class, or fixed answers. It does not inject the resolved
target of the `._print_helpers` relative import into the prompt.

## On AIScalar: update and test

```bash
cd ~/RAVE_FL_Research/AdaptiveContextOpt_AITECH
conda activate rave-fl
git pull --ff-only origin AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
```

## Option 1: Reuse the completed A/B/C experiment, run D only

Recommended for saving inference cost, using the exact same model and
seed as the existing output.

```bash
python -m aitech.ollama_localization \
  --model qwen3.6:27b \
  --seed 42 \
  --only dependency_aware_provenance \
  --output results/aitech/sympy20590_provenance_only.json

python -m aitech.compare_four \
  --previous results/aitech/sympy20590_ollama_three_arm.json \
  --provenance results/aitech/sympy20590_provenance_only.json \
  --output results/aitech/sympy20590_four_arm_comparison.json
```

## Option 2: Run all four anew in one experiment

```bash
python -m aitech.ollama_localization \
  --model qwen3.6:27b \
  --seed 42 \
  --output results/aitech/sympy20590_four_arm_fresh.json
```

The JSON includes predicted file, entity, correctness, selected IDs,
known missing dependencies, input/output Ollama token counts, and
wall-clock duration for each condition.

## Critical experimental controls

- A = all seven archived source lines, not full bug context.
- B and C = the same 36 regex-proxy-token budget.
- C and D = **identical selected evidence**, with D adding provenance.
- D has more prompt text, so C/D **are not total-input-token matched**.
  Use measured Ollama input tokens. Any D accuracy gain is evidence of
  the benefit of provenance **at an increased formatting cost**.
- Model, issue description, temperature 0, and seed 42 remain fixed.
- The user supplied A/B/C output from one execution. D-only is one new
  execution, not a full same-session replication.
- The original raw input `prompt_eval_count` was surprisingly large
  relative to prompt characters (likely model overhead); validate the
  Ollama model-template accounting separately.
- No new test execution, actual repo indexing, graph construction, or
  SWE-bench full-agent run happens here.
- The source line annotations are from archived tool output and not
  verified against a fresh checked-out repository. Validate that prior
  to claiming fully automatic source-aware retrieval.

**Decision gate:** If D fixes the file but costs substantially more
tokens, next implement compact provenance encoding and use the actual
model tokenizer to keep C/D within the same total prompt budget.
If D still predicts the wrong file, evaluate relative-import resolution
using real repository AST without leaking gold labels. Repeat on multiple
bugs, repositories, and seeds before making general claims.
