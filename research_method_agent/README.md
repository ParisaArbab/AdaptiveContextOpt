# Method Discovery Agent for GRAFT-FL / RAVE-FL

A reproducible small-bug experiment to **collect real independent LLM observations**, replay alternative evidence policies on the same candidates, and report matched Top-1 / Top-5 / MRR and token costs. This is a **research pilot**, not a claim of improved localization.

## What is tested

| Arm | Research idea | Actual implementation |
|---|---|---|
| `l1` | Existing cheap context baseline | L1 for every candidate, fixed prior/evidence fusion |
| `adaptive_k` | Taguchi et al. | Largest gap in initial localizer scores plus one buffer, then L1 on selected methods |
| `dependency_repair` | Hu et al. | L1 for all, then targeted L1 plus source of explicitly declared `depends_on` methods for at most two candidates |
| `risk_full` | RAVE-FL risk gate | L1 for all, Full verification of two most uncertain candidates |
| `esr_proxy` | Liu et al. inspired | Uncertainty × downstream call-link influence scheduling, **not** OECP's exact-verifier algorithm |
| `cascade` | Ashok et al. inspired | Cheap L1 model followed by expensive Full for uncertain candidates, optional distinct models |
| `code_cards` | Ranking-aware compact evidence idea | Deterministic source-line cards, **not** GistRank learned latent gist tokens |
| `llmlingua2` | Pan et al. | Actual LLMLingua-2 compression **of failure text only** when dependency installed |
| `combined` | Proposed DARC-FL pilot | All L1, selected dependency repairs, then risk-based Full verification |
| `conformal_k` | Calibrated candidate retention | Split-conformal threshold on baseline priors, requires separate method-gold calibration bugs |

**Important:** Code cards and ESR proxy are low-cost proxies. We do not claim to reproduce or have tested the original GistRank or OECP models. If an arm's real observation is missing, it is marked `missing_real_observation`, not assigned an invented score. The report never mixes file-level modified-class proxy labels with method-level gold.

## Prerequisites

Python 3.10+, access to your existing RAVE-FL preparation artifacts, and for new model evaluation an Ollama server with Qwen already installed. **No Defects4J checkout, Ollama, or GPU is available in the ChatGPT execution environment used to build this agent**, so the enclosed test results are synthetic software tests, not new benchmark measurements.

The working source and all tests use only Python's standard library. LLMLingua-2 is optional and needs `pip install llmlingua` plus its model weights. Do not install packages or download models on a shared server without checking quotas.

## Create a five-bug sample from real prepared RAVE-FL inputs

RAVE-FL `runner.py` writes one `instances/<safe-id>/input.json` with actual source candidates, L1 sketches and observed failure evidence. Prepare real Defects4J inputs with the **existing** repo's runner and immutable buggy checkouts. Do not substitute synthetic samples or post-fix code for real experiments.

```bash
python research_method_agent.py import-rave \
  --run-dir /absolute/path/to/rave_eval_dryrun \
  --max-bugs 5 --per-project 1 \
  --output pilot/manifest.json
```

For each candidate, the importer copies its real `levels[1]` (L1), `levels[2]` (Full), and original prior. For initial experiments, the script proposes `depends_on` based on unambiguous method names called in source. These are **heuristic links**, not guaranteed dependencies. Inspect them, or replace them with verified AST/call/data-flow edges before interpreting dependency preservation.

## Collect independent observations, no gold labels allowed

```bash
python research_method_agent.py collect \
  --manifest pilot/manifest.json \
  --observations pilot/observations.json \
  --output pilot \
  --ollama-url http://127.0.0.1:11434 \
  --l1-model qwen2.5:7b --full-model qwen2.5:7b \
  --max-bugs 5 --max-calls 150
```

This executes all candidate/method versions independently, including cheap L1, Full, dependency bundles (when a link exists), and small source cards. The optional `--llmlingua2` adds a real LLMLingua-2 failure-text compression arm if available. On an AIScalar machine with a separate Ollama port, change `--ollama-url` to the actual endpoint.

Observations are saved **after every real call**. Collection can resume with exactly the same manifest hash. Failures are logged and stop the run, and unknown token usage causes a hard error. `--max-calls` is a safety cap for new expensive calls. Physical collection costs more than a single policy because it evaluates all independent possibilities. The subsequent `evaluate` step uses **logical policy costs** from measured provider tokens. These are distinct quantities.

## Bring in OFFLINE gold and compare all methods

The repo's `evaluation_only.json` holds ground-truth labels, separate from inputs, and may have *file-level* labels but no method-level labels.

```bash
python research_method_agent.py import-gold \
  --run-dir /absolute/path/to/rave_eval_dryrun \
  --manifest pilot/manifest.json \
  --output pilot/gold_offline.json

python research_method_agent.py evaluate \
  --manifest pilot/manifest.json \
  --observations pilot/observations.json \
  --gold pilot/gold_offline.json \
  --budget 20000 --max-verify 2 \
  --output pilot/evaluation
```

Produces `results.json` (per-bug decisions and ranks), `summary.csv` (paired metrics), and `report.md` (readable comparison). Compares arms only on the **same complete bugs**. Incomplete arms are visible and excluded from paired means. Modified-class file labels must not be called method-level bug truth.

## Conformal calibration

To test `conformal_k`, add `"split": "calibration"` to **independent calibration bugs** and `"split": "evaluation"` to untouched evaluation bugs. This requires curated, complete **method-level** gold for calibration and an exchangeability assumption. With fewer than nine calibration bugs at alpha=0.10, the threshold may select all candidates, as it should. Do not describe its marginal candidate coverage as a guarantee of Top-5 localization or fixed token savings.

## A valid experiment and decision

Keep the candidate pool, evidence, model seed, output cap, final ranker, budget and gold semantics identical across arms. For the first run use five independent bugs across projects, then scale to 10 to 15 development bugs and a separate untouched evaluation set. Examine the per-bug wins **and** regressions, not only the mean. Do not select a winner from a one-bug Time-25 replay or a synthetic test fixture.

Acceptance to a larger test: better MRR at comparable provider tokens, or similar MRR with meaningfully fewer tokens; stable across independent bugs, with paired confidence intervals. A new paper-worthy claim needs substantially more than five bugs and a frozen held-out sample.

## Safety and reproducibility

- Only explicit pre-fix source and observed failure inputs go to the model. Gold is loaded after collection, in `evaluate` only.
- Recorded complete calls determine replay results. No made-up observations or fallback scores.
- Estimated admission happens before an action, and measured provider tokens are charged afterward. Any measured overrun stops further actions and is **not** reported as a valid budget-compliant result.
- Code cards preserve lines extracted from source but do **not** prove semantic completeness.
- `conformal_k` is not an unconditional guarantee. Gold missing from the pool is a candidate-pool failure.
- Do not infer speedup from shorter prompts; measure wall-clock end-to-end runtime on AIScalar separately.

## Quick self-test (no GPU)

```bash
python -m unittest discover -s tests -v
```

All example outputs shipped in the smoke-test folder are labeled **SYNTHETIC_TEST_ONLY**.