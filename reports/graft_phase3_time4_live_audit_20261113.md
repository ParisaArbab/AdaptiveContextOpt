# GRAFT-FL Phase 3, Time-4 real Qwen result audit

**Run:** `results/graft_phase3_test_aware_time4_20261113_210525`  
**User-reported experiment commit:** `6bc8bd2`  
**Status:** All 47 tests passed, the Time-4 run completed, all experiment artifacts were pushed. This audit uses saved real Qwen responses and adds **zero** new inference calls.

## Experiment and accounting

- **47 tests passed** in the offline audit, and **47 passed** again before the live run.
- Offline source-only audit selected `org.joda.time.Partial.with(DateTimeFieldType,int)` from saved valid L1 observations, not from gold.
- Both old focused and new test-aware evidence packages were ready under the **4,096 byte-proxy unit** input cap.
- Shared provider accounting over the run: **27 physical new Qwen calls**, **12,806 physical provider tokens**, `usage_unknown=false`, `pending=false`. This is not the standalone cost of each arm; request caching and shared results matter.
- All ten configurations completed. The comparative rankings were identical under each fusion.

## Gold ranking and logical token counts

| Policy | RRF gold rank | Suspicion-rank blend gold rank | Logical tokens |
|---|---:|---:|---:|
| fixed_l1 | 6 | 1 | 10,150 |
| l1_anchor | 6 | 1 | 10,150 |
| l1_graph_focused | 6 | 1 | 11,469 |
| l1_graph_test_aware_nostop | 6 | 1 | 11,487 |
| l1_graph_test_aware | 6 | 1 | 11,487 |

All graph policies made **two** graph calls for Time-4, so the opt-in stopping rule did not reduce calls on this case. This was expected: the top-ranked L1 candidate had a nonempty `missing_evidence` question (`null check in indexOf() method`), which deliberately blocks stopping. The with-stop and no-stop graph selections/responses are identical.

Relative to `l1_graph_focused`, the test-aware policy used **18 more logical tokens**, about **0.16%**, and did not improve rank. Relative to L1-only, it used 1,337 more logical tokens, about 13.17%. **Do not** sum per-arm logical tokens to infer provider cost.

## Model's graph judgments for the gold-labeled function

Gold-labeled method: `org.joda.time.Partial.with(DateTimeFieldType,int)`.

| Aspect | Focused graph | Test-aware graph |
|---|---|---|
| Suspicion | 0.2 | 0.1 |
| Uncertainty | 0.8 | 0.9 |
| Short explanation | `Index check missing in some cases` | `Test did not catch expected exception` |
| Missing evidence | `No failure in existing tests` | `Exception handling not tested` |
| Verdict applied to score | No | No |
| Guard reason | `unsupported_graph_downgrade` | `unsupported_graph_downgrade` |

Both responses reduced the original L1 suspicion of 0.9 and admitted missing evidence; the guard preserved the L1 score. Thus the Top-1 under `suspicion_rank_blend` remains a **fusion-only** effect on this development bug, not a graph-retrieval improvement.

## What changed in evidence

### Phase 2 focused

- Selector: `focused_compact_v1`.
- Package: **4,096 / 4,096** budget units.
- Included the target signature, `if (index == -1)` branch, and **full available source units** for `AbstractPartial.indexOf(DateTimeFieldType)`.
- Omitted post-branch setter and return nodes.
- Recorded incomplete target semantics.

### Phase 3 test-aware

- Selector: `test_aware_compact_v1`.
- Package: **4,071 / 4,096** budget units.
- Included more target code: `if (index == -1)`, `if (value == getValue(index))`, `getField(index).set(...)`, and `return new Partial(this, newValues)`.
- **Did not include** the `AbstractPartial.indexOf(DateTimeFieldType)` implementation, which was explicitly requested in L1 `missing_evidence`, and omitted `int[] newValues = getValues()`.
- Correctly reported `missing_callee_bodies`, `omitted_target_code_ids`, and `semantic_completeness_guaranteed=false`.
- The supplied test signal was `kind=possible_expected_exception`, `certainty=incomplete_test_excerpt`, `catch_type=null`, `evidence_origin=supplied_text:trigger_description`, `full_test_verified=false`.

**Scientific caution:** The Qwen claim `Test did not catch expected exception` is not established by the evidence. The saved trigger-description excerpt ends before any complete `catch` block and is not an observed complete execution trace. We can infer only that a `fail()` call after `with(...)` is compatible with an expectation of an exception. Do not label an omitted catch block as proof of absence.

## Conclusion

This is a **negative result for graph-specific localization accuracy** on the repeatedly studied development instance Time-4. The test-aware selector changed which real source fragments the model saw but did not improve the final rank or the model's uncertainty. The conservative stop rule was inactive because the model still had an open evidence question.

## Next scientifically useful experiment

1. Obtain the **authentic full failed test source** from the exact Time-4 buggy checkout, verify source path and SHA256, and distinguish it from original `trigger_description` evidence. Do not append invented exception handlers or extract answers from a fixed version.
2. Run a transparent test-expectation extraction on the complete test code, including whether a `catch` block exists and which exception it actually checks. Treat the test expectation as evidence, not as ground-truth buggy method identification.
3. Freeze a small **paired** budget-matched comparison of source-focused, test-aware-short, and test-aware-full-test inputs. Add a rule that directly checks whether missing dependent method bodies could invalidate the judgment.
4. Separately validate the stop gate on multiple representative, authentic bugs where the gate **activates**, and measure avoided calls and localization quality without accepting regression.
5. Recover real missing inputs for independent bugs before making claims of generalization. The prior Time-1/Lang-1/Math-1 manifest had no checkout or failure evidence in the queried locations.

No provider call or separate AIScalar execution was performed for this audit. All interpretation above derives from user-shared run output and saved GitHub artifacts.
