# Existing evidence before the method-discovery pilot

This is an archival baseline, **not a result produced by this new agent**. It uses the previously recorded actual GRAFT 100-bug study at `data/graft_100_cohort_20261115/aggregate/aggregate.md`, from the `graft-fl-recovery-analysis-20261116` branch.

| Fully paired held-out real-graph bugs | L1 file MRR | Focused graph MRR | Test-aware graph MRR |
|---|---:|---:|---:|
| 44 of 80, RRF fusion | 0.4948 | 0.4948 | 0.4948 |
| 44 of 80, suspicion-rank blend | 0.5320 | 0.5320 | 0.5320 |

Current graph evidence produced no mean file-level MRR gain on those matched runs. The report also records incomplete cases, so the 44 evaluated bugs are not a complete 80-bug sample. The archived collection executed 1,691 new Qwen calls, totaling 2,532,709 physical provider input-plus-output tokens. The graph policies averaged two graph calls on completed bugs, while the L1 policy used zero graph calls.

Research goal for the next pilot: determine whether targeted dependency restoration or uncertainty-controlled Full verification can **change the ranking in the right direction**, not simply add graph evidence. Use actual paired observations for that question. The existing aggregate is insufficient to simulate unseen per-method dependency prompt responses, and no accuracy values for new arms are inferred from it.

For method-level localization, the recorded GRAFT benchmark uses modified-class file proxy labels, not reliable method-level labels. Do not report these results as method MRR. The agent's independent L1 prompting and rank fusion are not identical to the original GRAFT L1 baseline.
