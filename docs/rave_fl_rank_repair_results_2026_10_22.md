# RAVE-FL V2: L1-first rank repair — 22 October 2026

The development change starts from the strongest observed cheap baseline: shared L1 batches for the whole pool. It then uses a capped, recomputed ranking-utility proxy to buy independent Full inspections. The frozen final ranker, candidate pool, full/L1 prompts and model generation settings were retained.

## Fresh inference results

Model: local `llama3.1:8b`, temperature 0, seed 0, 160 output tokens per candidate. One previously inspected bug: Time-25, 24 candidate functions. Gold is evaluation-only. The unmasked Full row is the audited historical reference; the other rows below are fresh inference. Masked evidence has its own freshly run Full reference.

| Evidence | Arm | Calls | Full calls | Provider tokens | Gold rank | Top-1 | Top-5 | MRR | Saved vs same-evidence Full |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| Original | Full reference (historical) | 24 | 24 | 21,378 | 2 | False | True | 0.500 | 0.0% |
| Original | fixed_batch | 6 | 0 | 11,298 | 4 | False | True | 0.250 | 47.2% |
| Original | repair_utility | 7 | 1 | 12,516 | 1 | True | True | 1.000 | 41.5% |
| Candidate names masked | full | 24 | 24 | 23,059 | 6 | False | False | 0.167 | 0.0% |
| Candidate names masked | fixed_batch | 6 | 0 | 11,710 | 23 | False | False | 0.043 | 49.2% |
| Candidate names masked | repair_utility | 6 | 0 | 11,710 | 23 | False | False | 0.043 | 49.2% |
| Masked + normalized numeric assertions (post-hoc) | full | 24 | 24 | 25,447 | 9 | False | False | 0.111 | 0.0% |
| Masked + normalized numeric assertions (post-hoc) | fixed_batch | 6 | 0 | 12,292 | 23 | False | False | 0.043 | 51.7% |
| Masked + normalized numeric assertions (post-hoc) | repair_utility | 6 | 0 | 12,292 | 23 | False | False | 0.043 | 51.7% |

Every fresh arm was independently audited against raw request/response ledgers, source/evidence hashes, provider counters and recomputed offline metrics. No skipped or invalid response is silently free. Replay and fresh inference are kept separate.

## Conditional replay: four selection controls

No new model calls were made for replay. It verified exact original independent prompts, original batch composition, model digest/settings, raw/parsed replies, actual costs and frozen ranking/tie-breaking. A selected response is revealed only after its action is chosen; preflight planning does not use future actual output cost.

| Policy at 16k budget, max 4 Full | Full calls | Accounted replay tokens | Gold rank | MRR |
|---|---:|---:|---:|---:|
| repair_top | 4 | 14,638 | 1 | 1.000 |
| repair_disagreement | 4 | 14,785 | 3 | 0.333 |
| repair_suspicion | 4 | 14,534 | 1 | 1.000 |
| repair_utility | 1 | 12,516 | 1 | 1.000 |

Fresh original inference exactly reproduced the replay's chosen calls, raw/parsed replies, token counters and final ranking: **True**.

The simpler baseline 'highest raw L1 suspicion, at most one Full' matched utility's cost and ranking: **True**. Therefore this bug does not demonstrate superiority or novelty of the utility formula. Baselines forced to spend four calls should not be used to imply otherwise.

The utility policy used the same one-Full solution for tested 14k, 16k, 18k and 20k budgets. At 12k, conservative whole-batch admission stopped before covering the complete pool. Recorded cheap cost below a budget does not mean the preflight bound will admit the same final batch.

## Action interactions and learning data

The offline dataset contains 282 conditional action rows from 13 distinct partial-repair states: 22 positive and 13 negative marginal MRR changes. All still belong to one independent bug. Future Full answers and actual costs are targets only; feature inputs use current evidence and estimated costs. No model was trained from these rows.

The oracle enumerated zero-to-four upgrades with gold and future observed costs. It is a diagnostic ceiling, not an online policy. Pair effects were nonadditive: a Full action can undo part of another action's ranking gain. The paired-state dataset preserves this dependence rather than summing singleton labels.

## Evidence ablation and limits

All candidate method names occurring inside test/frame method identifiers were replaced by opaque aliases, not just the gold method name. Assertion lines and expected/actual values were preserved exactly, together with scenario suffixes, class/file names and line numbers. This removes direct method-name cues, not every possible localization cue. The ablation uses the same bug, not an independent test case.

The original-to-masked result is a major failure of robustness: Full fell from rank 2 to 6, and L1/utility from rank 4/1 to 23. The utility found no positive proxy and did not upgrade anything. The positive original result therefore cannot be presented as name-independent fault reasoning. Since masked Full itself fails Top-5, this is also an evidence/model-grounding problem, not solely an allocation problem.

A separate exact-action replay of the masked condition still found an oracle one-Full repair reaching rank 1 at 12,999 accounted tokens. That gold-informed bound is not deployable: it shows that allocation opportunity exists, while the untrained proxy cannot recognize it from weak L1 evidence. Full verification of every method is therefore a reference, not an accuracy upper bound for mixed evidence.

One previously inspected development bug cannot establish accuracy preservation, statistical noninferiority, robustness across projects, calibration or paper novelty. The final utility is untrained, and its positive proxy/stop rule is not a guarantee of real MRR gain. The frozen ranker is the pilot fusion ranker, not a reproduced SieveFL final LLM ranker. Token totals cover verification only; local latency and monetary API savings are separate questions.

## Decision for the next research step

Continue with L1-first rank repair as the experimental architecture. Keep single-Full raw suspicion and small top-L1 escalation as strong controls. Do not return to unrestricted L0 -> L1 -> Full inspection of every candidate, and do not claim the more complicated proxy is necessary based on this result.

Before scaling, test evidence grounding and the simple controls on a small number of additional independent bugs. If those checks hold, freeze the policy/evidence recipe and collect about 10–15 independent exploratory bugs under shared candidate/evidence/model/ranker conditions. Preserve an untouched bug/repository-level test split. Only after observing reproducible marginal-action gains should a learned utility/risk model be trained and calibrated on separate bugs. Measure paired Top-1/5, full-pool MRR, tokens, coverage and violations; do not count states, seeds or evidence variants as independent bugs.

Related work already covers learned efficiency-aware localization control: [FuseSearch](https://aclanthology.org/2026.findings-acl.143/) uses SFT/RL for adaptive parallel execution. This pilot studies post-localizer verification allocation; empirical superiority and broader novelty remain open.

See [implementation and reproduction instructions](rave_fl_rank_repair.md), the saved protocol in `configs/rave_rank_repair_protocol.json`, replay artifacts under `results/rave_fl_rank_repair/replay_final`, and fresh ledgers under `results/rave_fl_rank_repair/live_original` / `live_masked`.

## Post-hoc numeric-evidence follow-up

After the masked failure was observed, a single further variant was specified in `configs/rave_rank_repair_posthoc_evidence.json`. A deterministic parser extracts the expected +04:00 and observed +03:00 offsets from the existing JUnit diff fragments and spells out their +3,600,000 ms difference. It retains the complete raw masked evidence, does not reconstruct ellipsized dates and adds no diagnosis or candidate identity. The policy, pool, model and limits are unchanged.

This is an explicitly post-hoc development experiment. The added text also changes length/repetition, so a difference is not proof of which formatting element caused it. Its fresh same-evidence Full/L1/utility results are in the table above. It is not an independent validation bug, and no further variant was selected to chase a favorable score.

Numeric clarification did not recover the masked failure: Full rank was 9, versus 6 without the added summary; fixed L1 and utility both ended at rank 23. The utility made zero Full repairs. Do not adopt this normalization as a proven improvement. The corresponding oracle still reaches rank 1 with one correct Full action, but the real policy does not find it.

## Verification

63 RAVE-FL tests passed, including 15 new rank-repair/replay/evidence tests. A broader scoped suite passed 83 tests after excluding the two previously documented files with unrelated failures. The three old failures were not modified or claimed fixed. Fresh experiment audits verified raw requests, responses, actual token totals and offline metrics in all three evidence conditions.
