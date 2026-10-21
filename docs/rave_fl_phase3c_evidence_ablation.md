# Phase 3c: Time-25 failure-evidence ablation (no ground-truth leakage)

## Controlled question

Does giving the **same** Qwen2.5 7B verifier more complete observed
failure evidence improve the ranking of Defects4J Time-25, and at what
token cost? This is **not** a new risk model or exploration policy.

## Fixed controls

- Actual buggy checkout `Time-25b`, not fixed source
- Same 24 candidate functions from SBIR/Ochiai/BoostN top-10 union
- Same method-source spans, verifier prompt, model `qwen2.5:7b`,
  temperature 0, selection threshold 0.60, and output cap 160
- Same token budget 60,000 *per arm*
- Same Ollama service at `127.0.0.1:11436` (Conda-based official
  Ollama, GPU 1 as configured on AIScalar)
- Ground truth is used **only after** predictions, for offline metrics

Note: deterministic temperature does not guarantee identical results
across runs or software versions. Record model/version and commit SHA.
The model returns separate `prompt_eval_count` and `eval_count`.

## Evidence A: historical baseline (already completed)

The original experiment used the real FlexFL trigger-test description
(492 characters, plus a newline from the loader), with no bug report.
Existing results must not be overwritten:

`results/rave_fl_phase3/expanded_ollama.json`

It verified 24 vs. 14 methods, measured 15,049 vs. 9,147 tokens, and
put the known faulty function at ranks 4 vs. 22. Full's 4th place was
a tie at suspicion 0.10, **not** a confident model diagnosis.

## Evidence B: reproduce the compressed observed failures

Defects4J `defects4j test` produces
`data/rave_fl_checkouts/Time-25/failing_tests`. It contains three
failing tests. The generator removes repeated JUnit and Ant frames and
keeps the observed assertion, test name, and up to three project frames.
No fixed-version source or `gt.json` is used for evidence.

```bash
conda activate rave-fl
cd ~/RAVE_FL_Research/AdaptiveContextOpt

python -m rave_fl.failure_evidence \
  --failing-tests data/rave_fl_checkouts/Time-25/failing_tests \
  --output results/rave_fl_phase3/time25_failure_evidence_rebuilt.txt

# Original user-generated evidence is 1,510 characters.
# Confirm the scripted generator matches it exactly.
cmp results/rave_fl_phase3/time25_failure_evidence.txt \
    results/rave_fl_phase3/time25_failure_evidence_rebuilt.txt

python -m unittest discover -s tests -p 'test_rave_fl*.py'
```

If `cmp` differs, inspect the diff before continuing. Do not silently
change the evidence while claiming a controlled experiment.

## Run evidence B

```bash
curl -sf http://127.0.0.1:11436/api/version

mkdir -p logs
nohup python -m rave_fl.compare_expanded \
  --flexfl-repo references/FlexFL_Time25 \
  --checkout data/rave_fl_checkouts/Time-25 \
  --bug Time-25 --top-per-method 10 \
  --evidence-file results/rave_fl_phase3/time25_failure_evidence.txt \
  --model qwen2.5:7b \
  --base-url http://127.0.0.1:11436 \
  --token-budget 60000 --max-output-tokens 160 \
  --output results/rave_fl_phase3/expanded_structured_evidence.json \
  > logs/time25_structured_evidence.log 2>&1 &
```

Allow it to finish before inspecting the JSON. This command runs two
arms, full and rank-threshold selective, using exactly the same evidence
file. The result records an evidence SHA-256, character count, and file
path. The 60k budget is approximate, not a hard token guarantee.

## Compare A versus B

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("results/rave_fl_phase3")
faulty = "org.joda.time.DateTimeZone.getOffsetFromLocal(long)"
for label, name in [("A original", "expanded_ollama.json"),
                    ("B structured", "expanded_structured_evidence.json")]:
    x = json.loads((root/name).read_text())
    print("\n", label, "evidence", x.get("evidence"))
    for arm_name in ("full", "selective"):
        a = x[arm_name]
        rank = a["ranked_functions"].index(faulty) + 1
        verdict = next((r["suspicion"] for r in a["calls"]
                        if r["method"] == faulty), None)
        print(arm_name, "rank", rank, "suspicion", verdict,
              "calls", a["verification_count"],
              "tokens", a["actual_tokens"],
              "top5", a["accuracy"]["top5"])
PY
```

This is a single-bug exploratory result. A rank improvement is not
statistical evidence of general accuracy gains. Test against other
Defects4J bugs before drawing research conclusions.
