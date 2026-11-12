# AITECH research report: Source-verified active retrieval and posthoc localization

**Date:** November 11, 2026
**Branch:** `AITECH`
**Scope:** CPU experiments on previously examined archived Defects4J evidence;
NOT an independent held-out AI fault-localization benchmark.

## Research hypothesis

An adaptive code-debugging agent should not automatically append every
dependency it discovers. Instead, it should decide whether to acquire
source code and whether the new evidence is worth its context cost,
preserving failure evidence and abstaining when the call binding or
diagnostic value is uncertain.

This is a **candidate research direction**, not a demonstrated new
state-of-the-art method. Graph retrieval, cost-aware RAG, bug anchors
and agentic fault localization are already studied in 2025-2026.

## Stage 1: Risk-gated expansion on three archived bugs

Compared rank fill, anchor first, unconditional sketch-call support,
and risk-gated support at the same four precomputed L1 sketch-token
budgets on Lang-1, Time-1 and Time-25.

- **12** bug/budget conditions, **5** context changes versus anchor-first.
- **0** graph-promoted choices.
- **0** anchor gains over anchor-first and **0** losses.
- Time-1 had **0** usable same-owner arity-sketch graph edges.
- No gold patch labels or LLM inference were used.

**Result:** Context differences came from packing order, not graph
expansion. The graph method is not validated by this experiment.

## Stage 2: Why graph coverage failed

Audited explicit class-qualified CALL records in frozen source sketches.

| Bug | Qualified calls | Matched candidate edges | Missing methods in ranked classes | Outside ranked classes |
|---|---:|---:|---:|---:|
| Lang-1 | 23 | 5 | 0 | 13 |
| Time-1 | 23 | 0 | 7 | 3 |
| Time-25 | 23 | 3 | 4 | 8 |

These are lexical hints, not compiler-verified binding counts.

## Stage 3: Source acquisition at the exact buggy revision

AITECH's failure-anchor-prioritized retrieval planner identified
`org.joda.time.DateTimeUtils.getChronology/1` as the highest-priority
missing method for Time-1, sourced from the archived failing constructor.

The Defects4J Time-1 buggy revision is
`8612f9e5b88c1bea933ef9ab1e431f5db3006b48`.
We fetched the method from that Joda-Time revision rather than main:

- `src/main/java/org/joda/time/DateTimeUtils.java`
- Lines **283-288**
- GitHub source blob SHA: `391cafd71d5a97fb4ce2d504010c98bff93b45cc`
- Method declaration: `public static final Chronology getChronology(Chronology chrono)`
- Source method size: **18 whitespace words**, **30 regex proxy tokens**.

The pinned method and provenance are frozen in
`data/aitech/time1_verified_dependency.json`.

The **source declaration** is verified, but the precise Java call
binding has NOT been compiler-attributed, and the retrieved method is
not automatically a bug-root-cause candidate.

## Stage 4: Matched proxy-budget ablation

Compared:
1. Anchor-first with original ranked evidence only.
2. Force the verified source method into the remaining context.
3. Gate the method until semantic binding is verified (abstain).

Both archived L1 snippets and retrieved source were charged with the
**same regex tokenizer**: word runs plus punctuation. Budgets:
300, 450, 600, 800 and 1,000 regex-token proxy units.

At every budget, forcing the helper retained exactly the same number
of failure-name anchors as anchor-first, though the included candidate
sets and total proxy usage sometimes changed.

## Stage 5: Post-selection changed-method recall

AFTER selections were frozen, we inspected the official Defects4J
Time-1 source patch:
`rjust/defects4j/framework/projects/Time/patches/1.src.patch`
(source blob SHA `c6dc26f8ddee82d52fc3a4066a24cec836a1f41d`).

Two patched candidate methods:
- `org.joda.time.Partial.Partial(DateTimeFieldType[],int[],Chronology)`
- `org.joda.time.field.UnsupportedDurationField.compareTo(DurationField)`

| Regex budget | Anchor-first | Forced historical helper | Gate unverified binding |
|---|---:|---:|---:|
| 300 | 1/2 | 1/2 | 1/2 |
| 450 | 1/2 | 1/2 | 1/2 |
| 600 | 1/2 | 1/2 | 1/2 |
| 800 | 2/2 | 2/2 | 2/2 |
| 1,000 | 2/2 | 2/2 | 2/2 |

**No changed-method recall improvement** from adding the historically
correct method in any tested condition.

This is one previously seen development bug, no statistical inference.
Changed-method retention measures evidence availability, not an LLM's
ability to localize the fault.

## Implementation / reproducibility

```bash
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.risk_gated_pilot --output results/aitech/risk_gated.json
python -m aitech.analyze_risk_gated_pilot \
  --input results/aitech/risk_gated.json --output-dir results/aitech/paired
python -m aitech.frontier_coverage_pilot \
  --output results/aitech/frontier_coverage.json
python -m aitech.frontier_request_pilot \
  --output results/aitech/frontier_requests.json
python -m aitech.time1_verified_retrieval_study \
  --output results/aitech/time1_verified_retrieval.json
python -m aitech.time1_posthoc_scoring \
  --output results/aitech/time1_posthoc_patch_recall.json
```

**GitHub CI:** Run `43074220917` passed **106 tests** and generated the
source-retrieval comparison and posthoc changed-method recall JSON.
GitHub Actions stores reproducibility artifacts in
`aitech-risk-gated-<run-id>`.

## Scientific novelty status

**Not yet demonstrated.** Cost-aware RAG, graph-based localization, and
agentic retrieval are established, including:

- Interprocedural graph localization and Bug Anchor Diffusion (TIC 2026)
  https://ieeexplore.ieee.org/abstract/document/11703025/
- Retrieval-Oriented Code Representations (2026)
  https://arxiv.org/abs/2607.11046
- BLAgent bounded agentic RAG (2026)
  https://arxiv.org/abs/2605.17965
- Cost-aware RAG (EMNLP Findings 2026)
  https://aclanthology.org/2026.findings-emnlp.413/
- CogniGent causal graph and hypothesis testing (2026)
  https://arxiv.org/abs/2601.12522

**Candidate narrower contribution:** a provenance-aware, failure-anchor-safe
**retrieve-or-abstain controller**, which explicitly estimates the
opportunity cost of fetching *missing* code and distinguishes verified
source definitions from compiler-verified bindings. Crucially, it must
be tested against strong code retrieval and agentic reranking baselines.

## Necessary evidence before a publishable localization claim

1. Verify requested call targets against complete pinned buggy checkouts
   with compiler symbol attribution, not merely class-name/arity matches.
2. Evaluate on more independent bugs and repositories. Reserve a genuinely
   unseen final test set. Freeze ranking and policy parameters on
   development data.
3. Measure how often retrieval helps *real changed-method recall*, not
   just lexical failure anchors. Include negative/degradation cases.
4. Use actual model tokenizer counts and retrieval+inference latency.
5. Run fixed-budget LLM localization on identical model, prompt and
   evidence pools and report MRR, Hit@1/5, patch correctness where possible.
6. Evaluate a cost-aware stopping policy against equal-budget rank fill,
   role-aware summaries, graph expansion and random retrieval controls.

## Scheduling status

The ChatGPT recurring research automation has been **disabled**, and the
daily cron event was removed from the AITECH research workflow at the
user's request. Research changes were committed interactively and
executed through push-triggered GitHub tests.
