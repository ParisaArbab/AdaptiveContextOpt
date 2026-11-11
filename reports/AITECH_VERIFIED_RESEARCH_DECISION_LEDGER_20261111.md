# AITECH Research Decision Ledger, verified 2026-11-11

**Purpose:** prevent apparent progress in scripts/green GitHub squares from
being confused with a *scientific result*. This report compares current work
with the **existing internal RAVE-FL V3 proposal** and verified AITECH pilots.

## 1. Research lineage and novelty caveat

The uploaded internal document *RAVE-FL V3: Risk-Adaptive Graph-RAG for Fault
Localization* (November 2026, six pages) already proposes a sequential
controller that chooses STOP, SEMANTIC, GRAPH, MORE_TESTS, and FULL actions,
with expected utility penalizing token, monetary, and latency costs.
It explicitly recommends calibrated risk prediction, graph provenance,
training/calibration/test splits and separate evaluation of patch outcomes.

Thus **a controller that chooses when to retrieve is not new to this project**.
Similarly, `reports/AITECH_PACES_FL_NOVELTY_HYPOTHESIS_20261111.md`
already proposes provenance-aware token allocation over L0, L1 and retrieved
definitions. The present work extends their implementation and auditing,
it does **not** establish the originality of that controller.

Related public 2025-2026 literature already has functionality-aware retrieval
(FaR-Loc), graph and agentic localization (ARISE, CogniGent, BLAgent),
multi-granularity semantic code search (CosFL), and code-representation
cost/quality experiments.

## 2. Actual completed CPU experiments, with outcome separation

### A. Three-bug selection, no model

Inputs: previously studied Defects4J Lang-1, Time-1, Time-25 source snapshots,
with separate posthoc official changed-method labels. Budgets 200, 300,
450, 600 *stored source-sketch proxy tokens*.

`aitech/failure_conditioned_study.py` compared rank fill, anchor-first,
relevance-density and optional lexical graph bonus:

- 12 paired bug/budget cases.
- 0 graph-bonus-induced context differences vs relevance-density.
- Time-25 failure-name anchor stayed present in all 4 budgets with
  anchor-first but not with rank-fill.
- No evidence for improved **LLM** localization or verified graph utility.

### B. Multi-resolution selector, changed-method L1 scoring

`aitech/multiresolution_study.py`, with separate posthoc scorer
`aitech/score_multiresolution_posthoc.py`:

- 12 bug/budget cases.
- Relative to the new single-L1 evidence baseline, one L1 changed-method
  coverage gain (Time-1 budget 200) and four losses
  (Time-25 budgets 200, 300, 450, 600).
- Lang-1 patched-method ID was kept at L0 but missing at L1 under all budgets.
- Method **identity** at L0 is not substantive method-body evidence.

### C. Gold-aware oracle ceiling, scoring ONLY

`aitech/changed_method_oracle_ceiling.py` shows limits of existing candidate
pool and fixed proxy budgets:

| Bug | Budget | Oracle possible L1 hits | New single-L1 hits | New tiered L1 hits |
|---|---:|---:|---:|---:|
| Lang-1 | 200 | 0 | 0 | 0 |
| Lang-1 | 300 | 0 | 0 | 0 |
| Lang-1 | 450 | 0 | 0 | 0 |
| Lang-1 | 600 | 1 | 0 | 0 |
| Time-1 | 200 | 1 | 0 | 1 |
| Time-1 | 300 | 1 | 1 | 1 |
| Time-1 | 450 | 1 | 1 | 1 |
| Time-1 | 600 | 1 | 1 | 1 |
| Time-25 | 200 | 1 | 1 | 0 |
| Time-25 | 300 | 1 | 1 | 0 |
| Time-25 | 450 | 1 | 1 | 0 |
| Time-25 | 600 | 1 | 1 | 0 |

This oracle **knows** the patch locations and is not a deployable
fault-localization method. It is a posthoc retrieval-availability ceiling.

### D. Critical strong-baseline control

`aitech/compare_strong_tiered_baselines.py` compared the proposed
multi-resolution policy against the strongest pre-existing AITECH tiered
strategies, including `anchor_rank_detail`:

**New method detailed-source wins: 0/12; ties: 8/12; losses: 4/12.**
Those losses were all Time-25 conditions. Therefore **reject this method
as the leading candidate**; do not write it as an improved method.

These studies, new regression tests, and posthoc scoring ran in successful
GitHub Actions evidence-selector workflow jobs. The labeled experiments
were NOT on a clean unseen test set.

## 3. Actual LLM testing is the next scientific bottleneck

A previous one-shot CPU model pilot exists at
`.github/workflows/aitech-cpu-llm-pilot.yml` for Time-1 and a verified
historical helper, but a completed model prediction artifact is not yet
confirmed in this ledger.

A new paired experiment has been implemented:

- `aitech/cpu_llm_localization.py`
- `.github/workflows/aitech-cpu-llm-paired-v2.yml`
- Open source `Qwen/Qwen2.5-Coder-0.5B-Instruct` on free public GitHub CPU.
- Actual model tokenizer and frozen identical failure contexts.
- Policies: rank-fill, established `anchor_rank_detail`, and experimental
  multi-resolution allocation, two previously examined bugs.
- Predictions and source hashes must be saved BEFORE patch labels are read.
- Logs and results are archived regardless of failure.

**This workflow is configured, NOT a verified completed LLM experiment.**
It must not be cited as accuracy evidence until model predictions and
scores are actually retrieved and inspected.

## 4. Research novelty: what would count

The internal proposal and public papers already discuss adaptive RAG,
code summaries and call graphs. The remaining plausible contribution is
not any one of those parts. A defensible new mechanism might be:

**Failure-conditioned, provenance-calibrated source upgrade/acquisition
with explicit expected harm and abstention**, evaluated on downstream
*model predictions* and potentially repair outcomes.

To establish it:
1. Extract and freeze initial candidates on many independent buggy
   repository revisions, keep provenance and real call-target confidence.
2. Collect paired model prediction/outcome changes from each evidence
   action under equal total token and inference budgets.
3. Train/calibrate risk model on separate projects, not three known bugs.
4. Compare to BLAgent/CosFL/FaR-Loc ideas, role-aware summaries,
   `anchor_rank_detail`, relevance-density and randomized acquisition.
5. Report method/file localization Hit@k and MRR, added method-detail
   retention, harmed-bug rate, token/latency costs, and patch success.
6. Demonstrate improvement on a genuinely untouched test set with
   confidence intervals and reproducible raw traces.

A small Qwen 0.5B CPU result alone cannot establish novelty, but is
a necessary engineering feasibility step.

## 5. Execution conditions

At the user's explicit request, the ChatGPT daily automation is disabled;
AITECH's recurring GitHub cron was removed. Code-push tests and one-shot
manual/push workflows are not recurring schedules. No AIScalar, no paid
inference API, and no artificial backdated commits.

**Overall decision:** No publishable localization improvement has been
demonstrated yet. Preserve negative evidence and move to independent
paired LLM experiments before training or claiming novelty.
