# AITECH candidate novelty: Provenance-Gated Marginal Action Value for Fault Localization

Date: 2026-11-11 | Branch: `AITECH`
Status: **implemented research mechanism and test harness, NOT proven novelty or
an empirically superior bug localizer.**

## 1. Falsifiable research hypothesis

Given the SAME failed-test text, code candidate pool, LLM, tokenizer,
decoding procedure and total prompt-token ceiling, can a model trained
on independently observed *paired before/after localization outcomes*
decide when to acquire more evidence, upgrade source detail or STOP?

Hypothesis: this decision reduces the fraction of **harmful** evidence
expansions and improves the localization-accuracy/cost frontier relative
to strong fixed selectors, while refusing unreliable code graph edges.

This is a testable hypothesis, NOT an established novel idea.

## 2. Proposed method

At debugging state `s` and candidate action `a`, estimate the
exclusive probabilities of improvement, no change and harm by fitting
a three-outcome classifier on independently observed paired LLM runs.

For each eligible evidence action, choose positive expected utility:

```text
U(s,a) = P(improve|s,a) * G
       - P(harm|s,a) * H
       - lambda * added_actual_model_tokens
       - mu * expected_added_latency
```

Eligible actions:

- `RETRIEVE`: acquire a new method/source evidence item
- `GRAPH`: request compiler-resolved dependency evidence
- `UPGRADE_L1`: signature (L0) to structural code sketch (L1)
- `UPGRADE_FULL`: add full code representation when actually available
- `STOP`: choose nothing whenever no action has positive utility

**Mandatory provenance rule:** `GRAPH` requires
`compiler_binding_verified=True`, regardless of user-supplied flags.
Compiler target identity is NOT by itself proof of causal sufficiency.
A separate independently audited evidence path is required to claim
support completeness.

**Mandatory calibration rule:** without a project-disjoint calibration
artifact, the runtime selects STOP. An uncalibrated heuristic is not
misrepresented as an outcome probability.

## 3. Implemented code and tests

| Component | File | What it does |
|---|---|---|
| Provenance-aware action selector | `aitech/marginal_action_policy.py` | Budget, binding and positive net-value gates; STOP |
| Leakage-safe feature schema | same file | Action cost, failure relevance, rank, semantic similarity, anchor and graph features |
| Paired outcome contract | `aitech/paired_action_outcomes.py` | Validates frozen raw prompt/prediction provenance, matched model/ceiling, project splits |
| Three-outcome training | `aitech/train_marginal_outcomes.py` | Optional scikit-learn multinomial logistic regression, scalar temperature selected on calibration projects only |
| Real paired Qwen test harness | `aitech/cpu_action_counterfactual.py` | Fixed candidates, STOP vs one L0-to-L1 upgrade, actual Qwen tokenizer and model, save raw output before scoring |
| One-time CPU workflow | `.github/workflows/aitech-cpu-action-counterfactual.yml` | Push-triggered once by workflow-file change and manual launch; no cron |
| Tests | `tests/test_aitech_marginal_action_policy.py` and `tests/test_aitech_paired_action_outcomes.py` | Budget, harm, unknown provenance, leakage, no-op, matched ceiling |
| Runner/trainer tests | `tests/test_aitech_cpu_action_counterfactual.py` and `tests/test_aitech_train_marginal_outcomes.py` | Frozen prompt pairs, independent outcome gating, untrained abstention |

## 4. What has and has NOT been measured

Engineering unit tests have executed in GitHub Actions. Existing
archived-selector experiments on three previously seen Defects4J bugs
have reproducible results, and they exposed harmful source-detail
downgrades in the earlier tiered candidate.

This iteration implements a **new LLM counterfactual runner**, but
**no completed raw paired Qwen inference artifact was verified at the
time this report was written**. We do not yet have a trained
project-separated marginal outcome model or measured improvement in
LLM localization accuracy.

The current one-time run covers only Lang-1 and Time-25, both
previously studied development bugs, with L0-to-L1 upgrade versus STOP.
The other actions are supported by the decision interface but not yet
by measured paired outcome experiments. The runner's posthoc method
label is a patch-changed-method proxy, not a verified causal root cause.

Training intentionally refuses missing or insufficient real outcome
data rather than fitting to fabricated labels. In particular, only
paired input JSONL from genuinely independent train/calibration/test
projects is eligible for training.

## 5. Important novelty threats from 2025-2026 literature

- FaR-Loc (2025), functionality-aware semantic candidate retrieval:
  https://arxiv.org/abs/2509.20552
- SHARE-RAG (2026), adaptive structural retrieval and hypothesis repair:
  https://doi.org/10.1016/j.jss.2026.113057
- HALT (EMNLP Findings 2026), verification-aware stopping:
  https://aclanthology.org/2026.findings-emnlp.343/
- AURA (EMNLP Findings 2026), learned utility-guided adaptive reasoning:
  https://aclanthology.org/2026.findings-emnlp.1617/
- ARISE (2026), repository-level graph and data-flow tools:
  https://github.com/FARD-Lab/ARISE
- Referential Dangling (2026), sufficient evidence support sets:
  https://github.com/JusperLee/Referential-Dangling

**Novelty cannot be claimed by combining these ideas.** We would
need to show a new, debugging-specific learned *outcome-dependent*
action policy using verified provenance, with a quantitative accuracy,
harm and cost advantage over those methods' relevant mechanisms.

## 6. Required controls and acceptance gates

1. **Freeze candidate pool, failing test and selection** before reading
   official patch labels. Save input SHA256, prompts, all generated raw
   responses, model revision, tokenizer and decoding.
2. **Paired decisions:** STOP vs each action (retrieve, graph, L1
   upgrade, full source) with same candidate identity where applicable,
   same actual tokenizer and same maximum total prompt tokens.
3. **Independent projects:** training, calibration and untouched final
   test split by project. Previously studied Lang-1, Time-1, Time-25
   must not be represented as unseen project-level evaluation.
4. **Calibrate P(improve), P(harm), P(unchanged):** multiclass Brier,
   log-loss and class confusion; include abstention coverage and
   harmful-acquisition rate. Check utility cost sensitivity.
5. **Strong baselines:** `rank_fill`, `anchor_rank_detail`,
   relevance-density, fixed L1, force-graph, stop-always and a
   cost-matched random action policy.
6. **Report ground-truth localization:** method/file Hit@1/5,
   MRR, changed-method sketch availability separately, actual model
   input/output tokens, inference and tool latency, selected evidence,
   missing gold from candidate pool and eventual patch success.
7. **Statistical evaluation:** paired per-bug comparisons on independent
   projects, bootstrap confidence intervals, harm analysis and strong
   failure cases. If no increase, report honest negative results.
8. **Data honesty:** do not treat sufficient path annotations, retained
   anchor names or compiler edges as proof of correct localization.

## 7. Commands and execution

```bash
git switch AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v

# Real CPU-only paired LLM experiment (downloads public model):
python -m aitech.cpu_action_counterfactual \
  --bugs Lang-1 Time-25 \
  --evidence-budget 650 --max-input-tokens 1800 \
  --output-dir results/aitech/cpu_action_counterfactual

# Train only after independently frozen and audited paired outcomes exist:
python -m aitech.train_marginal_outcomes \
  --pairs-jsonl data/aitech/audited_independent_action_pairs.jsonl \
  --output results/aitech/marginal_calibrated_model.json
```

The training input path above is a **future required dataset**, not a
file that currently exists. The code will fail explicitly if it is
missing or insufficient. No paid API, GPU or AIScalar is required.
No scheduled research automation or cron is enabled.

## Research decision

**GO for real paired action-outcome collection and provenance-calibrated
risk evaluation. NO-GO for claiming a novel or superior method yet.**
The smallest useful positive result would be a reduced harmful
expansion rate at matched model input budget AND no loss of held-out
fault localization. Even this requires independent replication.
