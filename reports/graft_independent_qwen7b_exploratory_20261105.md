# GRAFT-FL independent exploratory comparison, 2026-11-05

## Scope and provenance
This is an exploratory, **not held-out**, three-defect comparison using Qwen2.5 7B and two separately executed conditions. The numeric results below are transcribed from the AIScalar terminal output shared on 2026-11-05; the full raw run directory has not yet been committed to GitHub. Treat this file as a preliminary report, not a substitute for the immutable raw artifacts.

AIScalar run directory: `results/graft_independent_qwen7b_20261105_123254`

Preselected defects: Lang-1, Math-1, Time-1. Corrected buggy-source checkouts and SBIR/Ochiai/BoostN candidate unions. Lang-1 and Time-1 each had 24 candidates; Math-1 had 19. Evidence kind for all three: `trigger_description`, not observed failing-test execution. Batch size 1 (Qwen previously returned only c0 in three two-candidate batches per bug, making the earlier run incomplete).

## Observed results

| Bug | Prior rank | Fixed L1 | Highest suspicion | L1 graph control | L1 graph anchored | GRAFT adaptive control | GRAFT adaptive anchored |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lang-1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Math-1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Time-1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

For **both GRAFT adaptive conditions**, Top-1 = 2/3, Top-5 = 3/3, mean reciprocal rank = (1/2 + 1 + 1)/3 = 0.8333. No ranking improvement was observed on these defects, with no room for Top-5 improvement given the initial prior ranks.

| Run-level provider cost | Control | Anchored |
|---|---:|---:|
| Physical new calls | 84 | 84 |
| Physical provider tokens, input + output | 62,141 | 62,229 |
| Defects completed | 3/3 | 3/3 |
| Unknown provider tokens | 0 | 0 |

Anchored used 88 more physical provider tokens (approximately 0.14%); do **not** interpret these whole-run totals as per-policy token costs.

## Interpretation
Failure-name anchoring showed **no benefit** on this exploratory three-bug set. All correct methods were already ranked first or second by the original candidates. The earlier Time-25 development result (rank 10 to rank 2 after adding failure-aware anchoring) is a **development finding**, and tuning on that defect precludes treating it as independent validation.

Do not claim a statistically supported improvement, held-out generalization, coding-agent patch-resolution improvement, or meaningful token reduction. The experiment used trigger descriptions rather than observed Defects4J execution logs.

## Next research protocol
1. Preserve and inspect raw control/anchored artifacts before further code changes; do not overwrite this run.
2. Select harder defects with a *predeclared, outcome-independent* rule, e.g. a frozen list or independent chronological selection. Stratify by original rank only for descriptive analysis, and explicitly disclose any use of gold labels in stratification.
3. Run identical evidence and candidate construction in paired arms, with batch size 1 and controlled model digest, seed and budget.
4. Compare complete labeled-policy outcomes (Top-1, Top-5, MRR) and physical/logical token costs across several projects, and report all incompatible or incomplete cases.
5. Separately compare failure-name matching with non-name evidence signals, because explicit method names in test titles can make this heuristic artificially easy.

Raw artifact status: AIScalar run exists, but has **not** yet been uploaded to this branch. GitHub alone cannot access AIScalar files.
