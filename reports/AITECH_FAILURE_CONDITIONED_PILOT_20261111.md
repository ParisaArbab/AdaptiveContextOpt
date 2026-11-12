# AITECH research note: Failure-conditioned marginal utility, archival pilot

**Status:** deterministic, CPU-only selector-level study, not a fault-localization accuracy experiment.

## Research question

Under a fixed context budget, can a failure-conditioned selector preserve failure-related code without the all-or-nothing cost of dependency closure? Does a dependency-aware marginal reward change selection compared with relevance-only packing?

The predeclared selector lives at `aitech/failure_conditioned_selector.py`, and the runnable comparison is `python -m aitech.failure_conditioned_study`. It compares rank fill, anchor first, relevance density, and failure-conditioned graph utility at four budgets (200, 300, 450, 600) on archived Lang-1, Time-1, and Time-25 input snapshots.

**Split:** Lang-1 and Time-1 are development examples; Time-25 is a distinct evaluation bug, but the source project overlaps with Time-1 and the bug was previously inspected. It is not a clean unseen test set. Source was frozen before this comparison; no gold patches or downstream LLM predictions were accessed.

## Pre-LLM observed results

The counts below come from a deterministic evaluation of the archived inputs and policy formula. Each figure is **failure-name anchor retention**, *not* true bug-file or bug-function recall.

| Bug | Budget proxy | Rank fill | Anchor first | Relevance density | Failure graph |
|---|---:|---:|---:|---:|---:|
| Lang-1 | 200 | 1/2 | 1/2 | 1/2 | 1/2 |
| Lang-1 | 300 | 1/2 | 1/2 | 1/2 | 1/2 |
| Lang-1 | 450 | 1/2 | 1/2 | 1/2 | 1/2 |
| Lang-1 | 600 | 1/2 | 1/2 | 1/2 | 1/2 |
| Time-1 | 200 | 1/2 | 1/2 | 1/2 | 1/2 |
| Time-1 | 300 | 0/2 | 1/2 | 1/2 | 1/2 |
| Time-1 | 450 | 1/2 | 1/2 | 1/2 | 1/2 |
| Time-1 | 600 | 1/2 | 1/2 | 1/2 | 1/2 |
| Time-25 | 200 | 0/1 | 1/1 | 1/1 | 1/1 |
| Time-25 | 300 | 0/1 | 1/1 | 1/1 | 1/1 |
| Time-25 | 450 | 0/1 | 1/1 | 1/1 | 1/1 |
| Time-25 | 600 | 0/1 | 1/1 | 1/1 | 1/1 |

### Important negative findings

1. **No dependency lift**: `failure_graph` and `relevance_density` selected exactly the same methods at all 12 bug-budget combinations. The lexical dependency bonus did not change the evidence context in this sample. This is *not* evidence that graph retrieval improves debugging.
2. **Anchor overprotection can hurt**: Time-1, budget 600: `rank_fill` retained relevance proxy score about 3.282 using 595 sketch tokens, while `anchor_first` retained score 1.116 using 596. Anchors are weak proxies and an expensive method may consume almost the full budget.
3. **No generalization claim**: There are only three known bugs, including two from the same Joda-Time project. No LLM was called and no ground truth was scored.

### Research hypothesis to test next

**Failure-conditioned multi-resolution evidence budgeting:** preserve short signatures/structural sketches for potentially relevant but expensive methods, and spend extra tokens on full code only where failure traces and dependency evidence justify the upgrade. This might avoid the large opportunity cost of selecting entire functions under tight budgets. This hypothesis is not yet established as novel relative to existing adaptive RAG/context compression work.

## Literature that limits a novelty claim

- [FaR-Loc, 2025](https://arxiv.org/abs/2509.20552), functionality-aware retrieval and LLM reranking.
- [On the Role of Fault Localization Context for LLM-Based Program Repair, 2026](https://arxiv.org/abs/2604.05481), extra retrieval context can harm and useful context is conditional.
- [Rethinking Automated Program Repair, ESEM 2026](https://doi.org/10.4230/LIPIcs.ESEM.2026.40), faults, localization and LLM cost interact.

The proposed contribution must go beyond generic graph-aware RAG, using a falsifiable **conditional fidelity/graph expansion policy** and independent end-to-end fault-localization evaluation.
