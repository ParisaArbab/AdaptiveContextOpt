# RAVE-FL pilot results — 22 October 2026

Implemented and tested a closed-loop pilot with Java AST sketches, four actions, risk/state features, token-aware admission, prior-preserving shared ranking, raw call ledgers, paired evaluation, and offline LightGBM training/calibration tools.

## Measured experiment

One real bug: Defects4J Time-25. The same 24-candidate SBIR/Ochiai/BoostN union and 1,510-character structured failure evidence were used throughout. Model: local `llama3.1:8b`, temperature 0, seed 0, output cap 160 tokens per candidate. The source is the official buggy revision `c7a581e55fc988bd90fa4bb1b0acece5181b7c5f`. Observed failure evidence was reused from the earlier server run; the Java tests were not rerun locally.

This is an additional local-model experiment. It does not reproduce the supplied Qwen2.5/A100 numbers. Full-code prompts and the final shared pilot fusion ranker are identical across the new arms. The shared ranker is not the original SieveFL final ranker.

| Policy / controller | Budget | Calls | Model tokens | Bug rank | Top-5 | MRR | Tokens saved vs Full |
|---|---:|---:|---:|---:|---|---:|---:|
| Full | 60,000 | 24 | 21,378 | 2 | True | 0.500 | 0.0% |
| Rank gate >= 0.60; shared ranker | 60,000 | 14 | 12,887 | 9 | False | 0.111 | 39.7% |
| Fixed L1 batches | 60,000 | 6 | 11,298 | 4 | True | 0.250 | 47.2% |
| Confidence; generous budget | 60,000 | 53 | 50,015 | 2 | True | 0.500 | -134.0% |
| Initial staged adaptive; generous budget | 60,000 | 53 | 50,015 | 2 | True | 0.500 | -134.0% |
| Confidence; budget 16k | 16,000 | 17 | 14,867 | 13 | False | 0.077 | 30.5% |
| Final cost-aware adaptive; budget 16k | 16,000 | 17 | 14,717 | 11 | False | 0.091 | 31.2% |

Both Full runs returned the same ranking and 21,378 provider tokens. Actual usage includes every prompt and response in each arm, including discovery and escalation. Both completed comparisons passed the independent prompt/usage/metric audit. There were no invalid JSON responses in these runs. No inference arm exceeded its budget.

## What the results mean

- Fixed batching saved 47.2% of verification tokens and retained Top-5, while reducing MRR from 0.50 to 0.25. It did not preserve the full ranking quality.
- The initial staged controller inspected nearly everything at multiple levels: 53 calls and 50,015 tokens, with the same rank 2 as Full. It behaved like Confidence on this instance. This was an inefficient untrained policy.
- The final controller skips expansions whose risk-weighted planning cost is no better than going directly to Full. Under the 16k policy budget it spent 14,717 tokens (31.2% below Full), but the gold method finished at rank 11 and never received full verification. Confidence spent 14,867 and finished at rank 13. Both failed Top-5.
- The staged high-budget adaptive run and final low-budget run differ in both the cost gate and budget. Their difference cannot be attributed solely to one of those changes. The equal-budget Confidence/Adaptive comparison is the relevant final ablation.
- Candidate-pool and screened-gold recall are 1.0 for the adaptive arms because no candidate is deleted and every method gets discovery. This is not evidence of preserved diagnostic accuracy: the budget-limited gold method was only cheaply screened.
- The adaptive rule-based bootstrap has not demonstrated an advantage over fixed batching. This finding argues against scaling the present policy unchanged, and does not justify cancelling the whole research idea from one bug.

## Risk learning status

48 real cheap observations were reused from initial L0 discovery and independent L1 fixed batches, without new LLM calls. There are 0 candidate-false-negative harm labels and 20 positive marginal ranking-action harm labels. The latter compares leaving a candidate cheap with replacing it by its measured Full observation while holding other candidates cheap. All labels are offline; no label or Full outcome enters a cheap feature vector or prompt.

Only **one independent bug** is available. No real trained/calibrated risk artifact is claimed: training/calibration/test must use separate bugs. LightGBM fitting, isotonic calibration, target selection, artifact reload and leakage rejection were verified with explicitly synthetic fixtures. Candidate-harm and ranking-action-harm targets are separately selectable; they must not be interpreted as the same risk.

## Validation and limits

- 48 RAVE-FL tests passed after the final changes, covering adaptive behavior, tail candidates, malformed replies, paid token accounting, budget overruns, shared ranking, model training, calibration, disjoint bug splits and manifest-to-report execution.
- The broader project suite has three existing failures outside this change: one Agent4SR seen-entity formatting test, and two untracked dependency tests referencing the absent `scripts/run_pipeline.sh`. These files were not modified to hide the failures.
- This pilot uses verification-stage tokens and a deterministic shared fusion ranker. It makes no claim about total SieveFL pipeline costs, API billing, bandit/RL training, cross-repository generalization, or statistical noninferiority.
- Confidence intervals are omitted for one independent bug. Observed failure test names contain a useful naming cue; a masked-evidence ablation is still needed before a stronger research claim.
- An earlier interrupted preflight inference is preserved separately and excluded from the completed paired experiments. No workspace-wide billing total is claimed.

## Reproduce and continue

See [the adaptive pilot guide](rave_fl_adaptive_pilot.md) for exact CLI commands, dependencies, manifests, controlled data collection, training and audit steps. Next collect a few additional independent bugs, train on ranking-action harm where appropriate, calibrate on separate bugs and compare the learned policy against Fixed Batching and Confidence on held-out issues. Do not tune on Time-25 and call it held out.

Local artifacts:

- `results/rave_fl_pilot/time25_llama31_8b/run.json` and `audit.json`: initial five-arm comparison.
- `results/rave_fl_pilot/time25_llama31_budget16k/run.json` and `audit.json`: final budget-limited comparison.
- `results/rave_fl_pilot/time25_rank_risk_dataset.jsonl` and `risk_dataset_profile.json`: offline learning inputs and limits.
- Per-arm raw requests, responses and decision traces are in each run directory.
