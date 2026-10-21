# Phase 3, live Ollama verification (experimental)

The two arms call a **real Ollama model**, with the same candidates and bug
context. Ollama's `prompt_eval_count` and `eval_count` are stored per call.
Ground truth is loaded **after** both runs, solely for evaluation.

## 1. Inspect Time-25 ground truth

```bash
python -m rave_fl.diagnose_ground_truth \
  --flexfl-repo references/FlexFL_OriginalReplication --bug Time-25
```

If no candidate matches, the top-5 candidate recall is zero and re-ranking
alone cannot recover the faulty method. This diagnosis does not silently
change the matching rules.

## 2. Prepare real code

Provide a JSON file mapping each *exact* candidate ID from the diagnostic
to the actual Java method body in the **buggy checkout**, for example:

```json
{"org.example.C.method()": "public void method() { /* actual copied code */ }"}
```

All candidate IDs must have source; otherwise the runner fails clearly.
Do not fabricate source for experiments. Do not use the example above as data.

## 3. Start Ollama and run

```bash
ollama list
# If needed in another terminal: ollama serve
python -m rave_fl.compare_llm \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --bug Time-25 --method Ochiai \
  --sources /absolute/path/to/actual_method_sources.json \
  --model llama3:8b --token-budget 12000 \
  --output results/rave_fl_phase3/time25.json
```

Choose an installed model tag from `ollama list`.

## Important limits

- Source extraction is *not automated yet*; this runner requires a real
  source JSON mapping and rejects missing method bodies.
- Phase 3 still uses five candidates, not the desired expanded pool.
- Full verifies each candidate; selective verifies the high-ranking
  subset. This is a **rank heuristic**, not a trained risk predictor.
- Model-reported token usage is real, but the pre-call admission check
  uses an approximate estimate. One call can overshoot the total budget.
- Calls are sequential; baseline and selective arms each have their own
  equal budget. No benchmark measurements are claimed until you run it.
