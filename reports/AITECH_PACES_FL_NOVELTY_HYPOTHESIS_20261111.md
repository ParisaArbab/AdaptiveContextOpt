# AITECH research checkpoint: Risk-gated acquisition and tiered evidence

Date: 2026-11-11
Branch: AITECH
Status: exploratory development experiments, no independently demonstrated novelty.

## Central research hypothesis

For AI fault localization, the right optimization target is not merely the
number of code methods included in a prompt. A useful controller must allocate
limited model-context tokens among:

1. Failure text and trace anchors,
2. Identifiers and signatures of candidate methods (L0),
3. Structural sketches or source code (L1),
4. Optional definitions and verified dependencies outside the ranked pool.

The controller should estimate the marginal debugging value of *upgrading*
a method's evidence or *acquiring* a missing definition, and abstain when
confidence or opportunity cost is unfavorable.

Proposed name (working title): **PACES-FL**, Provenance-Aware Context
Evidence Selection for Fault Localization. This is an unvalidated proposed
method name and hypothesis, not a publication or established technique.

## Negative evidence that shaped the hypothesis

### 1. Graph expansion without coverage

Risk-gated vs anchor-first, three Defects4J archived real-bug candidate
pools, four L1 token-proxy budgets:
- 12 matched budget conditions
- 5 changed selected contexts
- 0 context choices promoted due to graph support
- 0 gains in failure-name anchor retention over anchor-first
- 0 improvements in posthoc changed-method recall

Interpretation: existing graph retrieval was too sparse, and changes
came from the packing rule, not proven graph utility.

### 2. Source retrieval without measured benefit

Time-1: identified and retrieved actual historic
`DateTimeUtils.getChronology(Chronology)` at buggy source
commit `8612f9e5b88c1bea933ef9ab1e431f5db3006b48`.
The method declaration exists, but semantic call binding is not confirmed.
Forcing this helper into evidence at matched regex token budgets
300/450/600/800/1000 gave **zero improved patch-method recall**.
This is not proof that the helper is useless for an LLM.

## Tiered source evidence experiment

For each method, archived metadata includes:
- L0 compact identifier/signature cost
- L1 lossy method source sketch cost

Five policies:
- `l1_only`: preserve full L1 when the whole sketch fits
- `tiered_l0`: maximize L0 method identity coverage
- `tiered_upgrade`: fill all affordable L0, then upgrade selected methods
- `anchor_detail_first`: preserve all failure-anchor L0 and upgrade
  anchor details before admitting additional method signatures
- `anchor_rank_detail`: same as anchor-detail-first, but then reserve
  detailed source for the highest-ranked non-anchor method before
  filling the remaining budget with optional L0 signatures

The rank and failure anchors are frozen from archived bug evidence
before looking at official patch labels.

### Posthoc changed-method evidence results

At proxy budgets 200, 300, 450, 600 on three previously examined bugs,
the `anchor_rank_detail` policy had:

| Bug | L1-only changed-method identity recall | Anchor+rank changed-method identity recall | L1-only detailed method recall | Anchor+rank detailed method recall |
|---|---|---|---|---|
| Lang-1 | 0/1 | 1/1 | 0/1 | 0/1 |
| Time-1 | 1/2 | 2/2 | 1/2 | 1/2 |
| Time-25 | 1/1 | 1/1 | 1/1 | 1/1 |

The same values occurred at each of the four fixed budgets in this
three-case exploratory development sample.

**Observed trade-off:** The new policy includes additional patched
method signatures in Lang-1 and Time-1, while retaining the same number
of patched L1 sketches as L1-only in this sample. This is an
identification/representation **coverage** result, NOT a measured
improvement in LLM fault-localization accuracy or patch repair.

The improvement is fragile to different rankings, anchor definitions,
token costs and bugs. All three bugs were studied earlier; there is
no independent held-out evaluation in this comparison.

## Novelty status and related work

Graph-based bug localization, adaptive retrieval, bounded agentic RAG,
and cost-aware RAG already exist. Strong claims require rigorous
comparisons to these baselines, not merely a new method name:

- Retrieval-Oriented Code Representations in Agentic Bug Localization:
  https://arxiv.org/abs/2607.11046
- BLAgent, bounded reranking for agentic bug localization:
  https://arxiv.org/abs/2605.17965
- Bug Anchor Diffusion and graph-based localization:
  https://ieeexplore.ieee.org/abstract/document/11703025/
- Cost-aware RAG (EMNLP Findings 2026):
  https://aclanthology.org/2026.findings-emnlp.413/
- CogniGent hypothesis/graph agent:
  https://arxiv.org/abs/2601.12522

The *potential* distinction is a single selective controller covering
source provenance, confidence in binding, multiple representation
levels, explicit abstention, and marginal cost of retrieval. Novelty
has **not been established** relative to these works.

## Independent validation required

A valid publishable evaluation will need:

- New and independent Defects4J projects/bugs split into development,
  calibration and genuinely held-out testing, no tuning on test bugs.
- Frozen complete buggy source checkouts, test failure texts,
  compiler-resolved call targets and source blob SHA-256.
- Independently scored gold patch-method Hit@1/5, MRR, top-k function
  recall and file recall, including misses when gold is not retrieved.
- Real model-token budgets including failure log, instructions, code
  evidence and generated output, not stored regex or L1 proxies.
- Identical code LLM and decoding, matched prompt instructions and
  seeds across conditions, plus repeated runs where relevant.
- Full L1-only, L0-only, anchor-only, random code expansion,
  always-graph expansion, risk-gated, and hierarchical upgrade baselines.
- Paired cost (retrieval calls, CPU/GPU wall time and inference tokens)
  vs localization quality with confidence intervals across bugs.
- An explicit stopping/abstention evaluation: when does acquiring
  another method actually improve posterior fault-localization accuracy?
- Negative cases, counterexamples, and error categories retained
  even when the proposed method underperforms.

## Current model experiment status

A separate no-schedule GitHub Actions one-shot workflow
`.github/workflows/aitech-cpu-llm-pilot.yml` is configured to
run the public 0.5B Qwen2.5-Coder model on CPU with two actual
tokenizer budgets. Predictions are frozen before patch-label scoring.

A successful completed model inference artifact has NOT yet been
verified through the connected tooling. **Do not cite LLM
localization gains from this setup.**

## Execution mode

At the user's request, ChatGPT's daily research automation has been
paused and the AITECH daily GitHub cron trigger removed. Research
changes here were implemented and tested interactively or on push.
No future scheduled or background research is promised.
