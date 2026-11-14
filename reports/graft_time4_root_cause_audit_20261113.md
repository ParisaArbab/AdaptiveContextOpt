# Time-4 GRAFT-FL root-cause audit, 2026-11-13

## Data provenance
Directly inspected committed, saved real Qwen2.5 7B outputs under `results/graft_time4_graph_ablation_20261113_143016/run/instances/Time-4_0d6a675374/` on this branch. This audit makes **zero new model calls**.

## Verified facts
- Correct method: `org.joda.time.Partial.with(DateTimeFieldType,int)`; its original candidate pool rank was **10**, and its final L1/graph/adaptive rank was **6**.
- Initial Qwen L1 observation for the correct method was **valid, suspicion 0.9, uncertainty 0.1**, with explanation `Method does not handle null DateTimeFieldType properly.` The missing evidence specified `null check in indexOf() method`.
- The correct method received **no graph inspection and no Full inspection**. Its L1 response is the only observation in its stage history for all three policies.
- Fused RRF score for correct method: **0.0306791569**. The first five candidate scores were approximately **0.0322581, 0.0317781, 0.0312576, 0.0310096, 0.0307692**, even though their latest suspicion values were 0.3 or 0.1. This demonstrates how original rank and reciprocal-rank fusion can outweigh one high L1 suspicion score.
- All 23 `failure_name_anchors` flags were **false**. The helper `failing_test_name_anchors` only processes `TEST:` lines and matches long, exact method-name tokens, and this evidence did not produce any match. The anchor bonus therefore had no effect.
- `l1_graph` graph calls went to `ZeroIsMaxDateTimeField.getMaximumValue(ReadablePartial,int[])` and `ZeroIsMaxDateTimeField.getMinimumValue()`; both returned suspicion 0.0 and reported missing/irrelevant evidence. Guard logic marked both responses **not applied to ranking**, so they did not alter the final scores.
- `graft_adaptive` inspected `Partial.Partial(ReadablePartial)` through graph and Full, and `Partial.getFieldType(int)` through graph. The correct method was not inspected in that policy either.

## Root causes in code
1. **The L1 graph selection policy ignores calculated action priority.** In `rave_fl/graft/controller.py`, `eligible` is selected with `min(... key=c.rank)` for `l1_graph`, rather than the calculated `priority`. This deterministically favors earlier FlexFL ranks and explains its two graph targets.
2. **The adaptive utility is heuristic, not calibrated to help the target rank.** It combines suspicion, uncertainty, score closeness, novelty and inverse predicted cost. Its selected targets had higher calculated priorities than the correct function. This is not necessarily a code bug, but a policy design problem.
3. **Original-rank RRF can dominate suspicion.** A valid Qwen suspicion 0.9 on the correct method was insufficient to move it into the Top-5.
4. **Failure-name anchor is inactive.** All flags were false, so this ablation does not test an *active* name bonus on Time-4.

## Controlled follow-up
Before any new inference:
- Add an offline alternative selection that picks highest **predeclared** utility for `l1_graph`, rather than original candidate rank, and unit-test its exact choices on a fixture. Report this as a **new policy**, not as an unchanged baseline.
- Compare frozen-score rerankings (RRF versus a gold-free fixed suspicion-weight alternative) and record all 23 ranks without choosing weights based on Time-4 ground truth. Do not claim result as independent validation.
- Evaluate anchor-match coverage across trigger-description vs observed failure-log evidence without injecting test names that are not present.
- Run paired, fixed-input ablations on new predeclared defects. Keep all invalid and unsuccessful outcomes; measure physical provider tokens separately from per-policy logical tokens.

**Finding:** The extra graph calls were real, but were directed to other candidates, and the two `l1_graph` updates were guarded out. The result gives a concrete evidence-selection diagnosis, not a general proof that graph context cannot help.
