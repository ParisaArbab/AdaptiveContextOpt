# AITECH: run the first three-arm SymPy localization comparison on AIScalar

## Already verified

The archived `sympy__sympy-20590` Agent4SR trace contains all seven
source anchors used by the AITECH microcase. At a 36-token **regex-proxy**
budget, both selectors used 35 proxy tokens. Relevance-only left one known
dependency dangling; dependency-aware left zero, preserving the import of
`Printable`. This is an evidence-level result, **not** improved localization.

The original historical RAW/LeanCTX agent predictions are recorded in
`results/swebench_sympy_20590/summary.json`; they are not AITECH outputs.

## Run the new downstream LLM comparison

On AIScalar (or a machine running Ollama):

```bash
cd ~/RAVE_FL_Research/AdaptiveContextOpt_AdaptivePilot
git fetch origin
git switch AITECH
git pull --ff-only origin AITECH
conda activate rave-fl

python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.sympy_trace_pilot --output results/aitech/local_trace_replay.json

ollama list
# If qwen3.6:27b is installed and Ollama is serving:
python -m aitech.ollama_localization \
  --model qwen3.6:27b \
  --output results/aitech/sympy20590_ollama_three_arm.json
```

If that exact repository directory does not exist, clone the AITECH
branch to a new folder and run from its root instead. This avoids
overwriting previous adaptive pilot work.

If the Ollama model is different, pass an **installed** model tag from
`ollama list`. All three arms use the same model, seed, temperature,
issue prompt, and evidence set.

The output includes predictions, file/entity hits, token-proxy counts,
Ollama prompt/generation token counts, and elapsed wall time.
Gold is evaluated **after** receiving the model response and is never
included in the prompt.

## Interpretation rules

- The three arms are RAW-EXCERPT (all seven extracted lines),
  RELEVANCE-ONLY (same 36-proxy-token budget as dependency-aware),
  and DEPENDENCY-AWARE.
- The RAW-EXCERPT arm is **not** the full original context.
- Measured Ollama input tokens may differ between arms even if the
  simple regex-proxy selected 35 tokens each, because content differs.
- Results are exploratory for **one manually selected source excerpt**
  from **one historical bug**. The methods do not yet automatically
  extract all failure evidence, build true graph dependencies, or
  reproduce the original SWE-bench issue through a fresh test execution.
- Do not claim increased bug localization accuracy without actual
  downstream LLM results and independent repeated bugs.

## Follow-up

Integrate the complete trace, graph extraction from AST/ARISE, source
provenance, and actual tokenizer budgeting. Run paired evaluations over
multiple independent bugs with repository-level train/test splits.
