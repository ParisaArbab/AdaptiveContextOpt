# AITECH: First Verified Real-LLM Paired Evidence-Action Study

**Experiment date:** November 11, 2026, New York time
**Branch:** `AITECH`
**Status:** Completed CPU model runs and validated training-record export.
**Scientific decision:** Real paired action outcomes exist, but there is
**no demonstrated localization improvement and no trained adaptive controller**.

## 1. Hypothesis under test

Can learning the *marginal benefit and harm* of acquiring debugging
evidence help an LLM decide when to retrieve, upgrade or STOP without
wasting input tokens and CPU time?

The current test isolates one action, `UPGRADE_L1`, a change from
method identity/signature to a saved structural code excerpt. Both
conditions use the **same frozen method identities, failing test, model,
decoding, output candidate IDs and total prompt-token ceiling**.

No graph expansion, semantic RAG retrieval or sequential controller
was actually evaluated in these real-model runs.

## 2. Model and reproducibility

- Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Pinned model revision: `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`
- CPU float32, greedy generation, manual seed 42, max 24 new tokens.
- Token costs use the **actual model tokenizer**, not regex proxies.
- Both prompts are frozen, raw predictions saved, then official Defects4J
  patch/changed-class labels are opened for scoring.
- Earlier strict `INDEX: N` responses failed with the small model.
  Final runs accept a single unambiguous numeric candidate index.
- The STOP baseline is restricted to at most eight L0 candidate methods.
  This ensures an optional upgrade can fit without changing candidate
  identities. This is fixed *before* reading official gold labels.

## 3. Real STOP versus L1-upgrade outcomes

| Bug | Scoring scope | STOP hit | L1-upgrade hit | Input tokens STOP -> upgrade | Added input tokens | CPU generation seconds STOP -> upgrade |
|---|---|---|---|---|---:|---|
| Lang-1 | changed method Hit@1 | Yes | Yes | 604 -> 629 | +25 | 2.681 -> 2.770 |
| Time-25 | changed method Hit@1 | No | No | 950 -> 956 | +6 | 4.612 -> 4.617 |
| Math-1 | changed class Hit@1 | Yes | Yes | 1087 -> 1386 | +299 | 5.582 -> 7.106 |
| Chart-1 | changed class Hit@1 | No | No | 1119 -> 1268 | +149 | 5.916 -> 6.379 |

**All four actions were neutral:** the predicted method identity did
not change in any pair. Total extra prompt tokens were **479** across
the four pairs. Measured CPU generation time also increased in aggregate,
but timings are not repeated controlled latency estimates.

**Critical: do not pool method-level and file/class-level Hit@1 into
one accuracy statistic.** Lang/Time are previously studied development
bugs; Math/Chart were collected from fresh buggy checkouts but are
already examined and therefore cannot be used later as untouched
final test cases.

### What the new Math/Chart data demonstrates

- Real buggy revisions of Math-1 and Chart-1 were checked out using
  the pinned Defects4J checkout framework.
- The genuine `failing_tests` file was read after executing tests.
- 25 pre-gold source method candidates were frozen for each bug.
- Source diversity was enforced: at most three candidate methods
  from the same file.
- At least one official modified class was present among the 25
  source candidates for each bug.
- The same small Qwen model produced valid numeric choices for STOP
  and UPGRADE_L1.
- The extra L1 code did **not** improve changed-class Hit@1 on either bug.

On Math-1 the prediction was a method of `BigFraction` in both arms
and the official modified classes include `BigFraction` and `Fraction`.
On Chart-1, Qwen selected a method of `StackedAreaRenderer` in both
arms while the official modified class is
`AbstractCategoryItemRenderer`.

These are **two individual bugs**, not statistical evidence that
more code generally hurts model accuracy.

## 4. Separate six-condition selector control

A separate same-Qwen development experiment compared `rank_fill`,
`anchor_rank_detail`, and `tiered_marginal` on Lang-1 and Time-25:

| Bug | rank_fill | anchor_rank_detail | tiered_marginal |
|---|---|---|---|
| Lang-1 changed-method Hit@1 | 0 | 1 | 1 |
| Time-25 changed-method Hit@1 | 0 | 0 | 0 |

All six model replies were parsed successfully in the corrected run.
These two previously examined bugs **cannot** establish generalization
or new state-of-the-art performance.

## 5. Validated train/calibration export

Actual Math/Chart paired outcomes were reloaded from the saved GitHub
Actions artifacts and converted into two validator-approved JSONL rows:

- **Math-1:** project `Math`, split `train`, outcome `0` (neutral)
- **Chart-1:** project `Chart`, split `calibration`, outcome `0` (neutral)

Both rows use the same frozen model revision, greedy decoding protocol,
`defects4j_changed_class_hit_at_1` outcome definition and total model
input-token ceiling 1800. They retain source, prompt, prediction and
failure SHA fingerprints. The converted dataset fingerprint is:

`39e1ada9c915577202925282d8b9a2ea3a5979a8ebbee5a26e4aeda0c3a77708`

The dataset is **not yet trainable** under AITECH's strict policy:
there is only one train project, one calibration project,
**no independent test project**, no positive outcome and no negative
outcome. The 3-class classifier deliberately rejects this case.

## 6. Real experiment artifact references

- Live Math/Chart source snapshots, raw replies and posthoc file scoring:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43141574195
- Development STOP versus UPGRADE Qwen run:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43141543711
- Six-condition real Qwen selector comparison:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43141606693
- Validated raw-outcome data export:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43142486515

No model-prediction result was invented. Initial failed/skipped studies
were retained in GitHub history, and corrections were implemented as
new commits using real dates.

## 7. Methodological limitations and next scientific gate

1. **Very small model:** Qwen 0.5B may show position bias. All four
   STOP/UPGRADE pairs selected the same numbered candidate; candidate
   position sensitivity must be measured.
2. **Only one action per bug:** the paired experiment did not yet
   compare multiple upgrade targets, semantic retrieval, verified
   graph-slice acquisition, or additional tests.
3. **Only two fresh projects:** collect multiple independently frozen
   bug cases from more project families, and keep future projects
   completely untouched until the final evaluation.
4. **No positive/harm labels:** the classifier must not train on two
   neutral rows. Collect actual before/after model outcomes for
   more actions; include deliberate STOP and strong fixed baselines.
5. **Candidate extraction limitations:** the Java regex/brace scanner
   is not a compiler-verified method extractor, and selecting from
   only 25 methods changes the localization task into a constrained
   candidate-choice task.
6. **No causal support claim:** javac call binding, code relevance
   and modified-class membership are not proofs of fault causality.
7. **Unmeasured retrieval cost:** CPU generation latency was measured
   but actual source-retrieval/tool latency and amortized token cost
   still require accounting.

**Decision:** continue with position-order controls and multiple
candidate actions per bug before training. Require independent
project-level train/calibration/test splits and positive, harmful and
neutral real-model outcomes. Do not claim novelty or model improvement
until robust matched-budget held-out experiments demonstrate it.
