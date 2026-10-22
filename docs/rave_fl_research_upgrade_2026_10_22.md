# RAVE-FL research upgrade, 22 October 2026

This note describes the **new executable options**, not claimed performance improvements.
The original Time-25 experiments and their outputs remain unchanged.

## 1. Live base localizer with genuine scores

Default pilot behavior remains the old precomputed SBIR/Ochiai/BoostN CSV union. Those rank priors are **not** semantic similarity or coverage measurements.

To use a real external base localizer, add `localizer_command` to each manifest case:

```json
{
  "cases": [
    {
      "bug": "Time-25",
      "project": "Time",
      "flexfl_repo": "references/FlexFL_Time25",
      "checkout": "data/rave_fl_checkouts/Time-25",
      "evidence_file": "results/rave_fl_phase3/time25_failure_evidence.txt",
      "localizer_command": [
        "python", "-m", "your_actual_localizer",
        "--bug", "{bug}", "--checkout", "{checkout}",
        "--failure-evidence", "{evidence_file}"
      ]
    }
  ]
}
```

The named `your_actual_localizer` module is a **placeholder**, not a bundled SieveFL implementation. It must already exist and run in your environment. This bridge executes the command once for the bug before any experiment arms, and records its invocation and candidate/source provenance. The command must output **one JSON object** to standard output:

```json
{
  "localizer": "your-localizer-version",
  "runtime_pruning": "not_available",
  "candidates": [
    {
      "function_id": "org.example.Foo.bar(int)",
      "file_class": "org.example.Foo",
      "start_line": 80,
      "end_line": 102,
      "semantic_score": 0.70,
      "coverage_score": 0.90,
      "ranker_scores": {
        "retrieval": 0.70,
        "coverage": 0.90
      }
    }
  ]
}
```

Every score must be finite and between 0 and 1 and derived from the real localizer, never copied from the offline gold labels. The source spans must match the **buggy** checkout. Set `runtime_pruning` to `measured` only when that upstream localizer actually measures runtime pruning. A field label alone is not independent proof that retrieval or coverage was measured.

The new source contract lives in `rave_fl/localizer_bridge.py`. It makes integration possible, but **does not bundle an original SieveFL model** or automatically generate dynamic coverage.

## 2. Evidence-based rank repair

`rave_fl/code_sketch.py` now records additional source-grounding signals:

- `body_failure_relevance`, lexical overlap of the supplied method body and observed failure
- `numeric_evidence_overlap`, overlap of meaningful numeric literals in source and observed failure
- `trace_class_match`, class-name presence in the failure evidence

These are cheap, **imperfect evidence proxies**. None confirms fault causation. They are separately visible even when the failing test contains no function-name cue.

Try the **new, opt-in** `repair_grounded` policy alongside the frozen older arms:

```bash
python -m rave_fl.pilot \
  --manifest configs/rave_pilot_time25.json \
  --model llama3.1:8b \
  --arms full,fixed_batch,repair_utility,repair_grounded \
  --token-budget 60000 --policy-budget 16000 \
  --max-full-repairs 4 --max-output-tokens 160 \
  --output results/rave_fl_grounded_new
```

This policy first pays for L1 screening of the whole pool, then selects Full inspections using a weighted **untrained** evidence/uncertainty proxy and the previous estimated ranking gain. Unlike `repair_utility`, a zero ranking-gain prediction does not automatically prohibit all Full calls. It can also spend more tokens without helping localization. It needs new independent experiments; do not claim it solves the masked-name failure before testing.

**Compatibility:** The feature schema of `train_risk.py` has grown. Old risk JSONL rows will fail strict validation until they are **regenerated from actual experiments**, preserving their provenance. Do not fill new features with invented values.

## 3. Multiple real bugs and trained risk

Prepare a study manifest with at least nine unique Defects4J bugs from **three or more independent projects**. Give each case an explicit `project` key and the real `checkout` and `flexfl_repo` paths.

Freeze a project-held-out split **before** training:

```bash
python -m rave_fl.study \
  --manifest configs/study_real_bugs.json \
  --out configs/rave_real_split.json \
  --metadata results/rave_study/split_metadata.json \
  --seed 0
```

Run the pilot on the actual bug set, increasing the per-run guard `--max-bugs`:

```bash
python -m rave_fl.pilot \
  --manifest configs/study_real_bugs.json \
  --model llama3.1:8b \
  --arms full,fixed_batch,confidence,adaptive,repair_utility,repair_grounded \
  --collect-risk --max-bugs 30 \
  --token-budget 60000 --max-output-tokens 160 \
  --output results/rave_study/real_data
```

`--collect-risk` adds independent L0/L1 probes and requires a complete Full reference; it uses more tokens than an evaluation-only run. Evaluate on a **separate, fresh held-out experiment**, not by running and training from the same outcomes.

If the recorded labels contain enough harmful and safe observations **in each partition**, fit the existing LightGBM + isotonic risk estimator:

```bash
python -m rave_fl.train_risk \
  --data results/rave_study/real_data/risk_dataset.jsonl \
  --split configs/rave_real_split.json \
  --target ranking \
  --output results/rave_study/risk_model.json
```

Only then rerun on a genuinely held-out bug manifest with `--risk-model results/rave_study/risk_model.json`; the loader rejects train/calibration bug overlap. **No such newly trained risk model or multi-project accuracy result was produced by this code update.**

## 4. Contextual bandit, guarded offline preparation

`rave_fl/bandit.py` implements per-action reward models with an upper-confidence selection rule and **self-normalized importance sampling (SNIPS)** for off-policy evaluation.

It is **offline only** and never silently replaces the default adaptive policy. It requires:
- Real **randomized** action logs, observed rewards, and known positive logging propensities
- The entire current feature schema and a consistent, versioned reward definition
- At least **10 distinct training bugs, 5 calibration bugs, and 5 held-out test bugs**
- Disjoint repository/project partitions, support for every action in training, and enough effective off-policy samples

The current deterministic Time-25 replay/counterfactual action rows **cannot** be passed off as randomized bandit interactions. To train when compliant logs exist:

```bash
python -m rave_fl.bandit \
  --data results/rave_study/real_randomized_actions.jsonl \
  --split configs/rave_real_split.json \
  --output results/rave_study/offline_bandit.json
```

The resulting artifact explicitly says `online_enabled: false`. Integration into online RAVE-FL requires prospective safety checks and a real randomized-data collection protocol; it is **not** completed by this update.

## 5. Multi-bug metrics, time and price

The pilot already compares paired Top-1, Top-5, MRR and provider input/output tokens. The upgraded report also exposes **local LLM-call seconds** and their paired difference.

If you want a hypothetical pay-per-token comparison, you may provide **your own** prices:

```bash
python -m rave_fl.pilot \
  --manifest configs/study_real_bugs.json --max-bugs 30 \
  --model llama3.1:8b --arms full,fixed_batch,repair_grounded \
  --token-budget 60000 --output results/rave_study/price_scenario \
  --estimated-input-price-per-million 1.00 \
  --estimated-output-price-per-million 2.00
```

Those two numbers are **illustrative user-supplied rates**, not advertised pricing or billed Ollama cost. The value excludes upstream retrieval, checkout, localizer run time and hardware rental. Measure those separately for a complete end-to-end benchmark.

## Validation

- Python unit tests: `tests/test_rave_fl_extensions.py` plus the existing RAVE-FL tests
- GitHub Actions: `.github/workflows/rave-fl-pilot.yml` on `codex/rave-fl-adaptive-pilot`
- Existing historical Time-25 results are untouched.
- No new real model benchmark was claimed or run as part of writing these modules.
