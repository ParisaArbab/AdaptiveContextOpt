# Phase 3b: expanded candidates + automatic Java source extraction

This work keeps all modifications on `feature/rave-fl-mvp`.

## Why expand the pool?

For Defects4J Time-25, the documented faulty function is
`org.joda.time.DateTimeZone.getOffsetFromLocal(long)`. The
existing five-candidate Ochiai example cannot recover it.

In the published FlexFL CSV rankings, it appears at **SBIR rank 9,
BoostN rank 9, and Ochiai rank 53**. The top-10 union of SBIR, Ochiai,
and BoostN can therefore contain the faulty function without inserting
it using ground truth.

## 1. Update and validate without Ollama

```bash
cd ~/Downloads/AdaptiveContextOpt
git pull origin feature/rave-fl-mvp
python -m unittest discover -s tests -p 'test_rave_fl*.py'

python -m rave_fl.expanded_pool \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --bug Time-25 --top-per-method 10 --diagnose
```

The `--diagnose` flag reads labels **only** for offline candidate recall.
The candidate construction and policies do not receive labels.

## 2. Get actual buggy Java source

This step requires `defects4j`, Java, and the checkout dependencies.
Use the project's supported server if your Mac does not have Defects4J.

```bash
command -v defects4j
mkdir -p data/rave_fl_checkouts
defects4j checkout -p Time -v 25b -w "$PWD/data/rave_fl_checkouts/Time-25"
```

Source extraction uses the exact `File`, `StartLine`, and `EndLine`
columns from the real candidate CSVs. For inner classes, it resolves
the enclosing `.java` file. It fails on missing/ambiguous files or
when the extracted line span does not contain the method name.
Do not run on fixed checkout `25f`.

## 3. Validate real method extraction, no model needed

```bash
python -m rave_fl.compare_expanded \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --checkout data/rave_fl_checkouts/Time-25 \
  --bug Time-25 --top-per-method 10 --dry-run \
  --output results/rave_fl_phase3/expanded_dry_run.json
```

If source lines mismatch the checkout revision, the script stops instead
of silently attaching unrelated code. Inspect and fix mapping before use.

## 4. Run with an actual Ollama model

Install Ollama from https://ollama.com/download and start the app/service.
Choose a model that fits available memory and appears in `ollama list`.

```bash
ollama list
ollama pull qwen2.5-coder:7b
python -m rave_fl.compare_expanded \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --checkout data/rave_fl_checkouts/Time-25 \
  --bug Time-25 --top-per-method 10 \
  --model qwen2.5-coder:7b \
  --token-budget 16000 --max-output-tokens 96 \
  --output results/rave_fl_phase3/expanded_ollama.json
```

The full arm verifies as many candidates as its budget admits.
The selective arm uses the *existing* rank-threshold policy; this is
a baseline for developing learned risk allocation, not a final RAVE-FL
policy. Both arms use exactly the same candidate pool and bug context.

Ollama's reported `prompt_eval_count` and `eval_count` determine
actual token usage. The admission estimate is NOT an absolute hard cap;
a call may exceed the requested remaining budget. Confirm
`budget_exceeded` in the output and treat such runs cautiously.

## Limits

- Time-25 is one bug and does not establish statistically meaningful
  token savings or fault-localization improvement.
- A larger pool is necessary but not sufficient. A rank-threshold
  selective policy may still fail to inspect the true faulty method.
- Source ranges depend on the benchmark CSV matching the actual buggy
  checkout. Validate snippets before interpreting outputs.
- The risk model is still a rank heuristic. Calibration, trained risk,
  uncertainty-based exploration, and rigorous model token limits remain.
- Verify `defects4j` and the actual Ollama executable are installed
  where the experiment will run, Mac or AIScalar.
