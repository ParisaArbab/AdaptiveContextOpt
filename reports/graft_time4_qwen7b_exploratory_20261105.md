# Time-4 exploratory paired Qwen2.5 7B result, 2026-11-05

This is a user-reported AIScalar result transcribed from terminal output, not yet a verified raw-artifact GitHub archive. It is exploratory, and Time-4 was selected after an offline, gold-aware difficulty screen. Do not call it held-out.

- Recorded Time-4 buggy checkout SHA: `bcb044669b4d1f8d334861ccbd169924d6ef3b54`.
- Candidate pool: 23 methods from SBIR/Ochiai/BoostN top-10 union, correct method originally ranked 10.
- Evidence: external FlexFL `trigger_description`, not an independently observed test failure.
- Same Qwen2.5 7B model, single-candidate batches, budgets, and candidates in both arms.
- Both runs completed with 29 physical new model calls each.

| Policy | Control rank | Anchored rank | Control Top-5 | Anchored Top-5 |
|---|---:|---:|---:|---:|
| Prior | 10 | 10 | 0 | 0 |
| Fixed L1 | 6 | 5 | 0 | 1 |
| Highest suspicion | 10 | 10 | 0 | 0 |
| L1 graph | 6 | 5 | 0 | 1 |
| GRAFT adaptive | 6 | 5 | 0 | 1 |

GRAFT adaptive MRR improves from 1/6 = 0.1667 to 1/5 = 0.2000, 20% relative, on this one bug.

| Physical run accounting | Control | Anchored |
|---|---:|---:|
| Qwen calls | 29 | 29 |
| Provider input+output tokens | 15,662 | 15,614 |
| Unknown usage | 0 | 0 |

The observed difference is 48 provider tokens (~0.31%) between independently executed whole-run conditions, **not** proof of token reduction by the method. Fixed L1 and L1 graph also improved by one rank, so the result does not isolate a benefit from graph refinement. A graph-specific ablation and more predeclared independent bugs are necessary.

AIScalar directory, not automatically accessible to GitHub: `results/graft_time4_qwen7b/`. Preserve and archive the raw manifests, per-arm results, reports, logs and accounting separately. Do not overwrite these runs.
