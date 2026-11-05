# GRAFT local real-model smoke results

Fresh local Llama3.1 8B development smoke; this is not an A100 measurement or independent held-out validation. Existing Time-1/Lang-1/Math-1 inputs were selected for availability before inference; the separate Lang-54/Math-80/Closure-61 protocol remains unexecuted.

The evidence is supplied trigger-test text, not newly generated observed failure logs. Official benchmark tests were not run. Math-1 source-span incompatibility is retained rather than replaced. All arms use RRF, the same pool, 8,192 context, 160 output tokens per candidate, batch size two and a 24K logical verification budget. Physical caps: 80K run / 40K per bug / 80 new requests.

| Bug | Policy | Status | Strict gold rank | Verification tokens | Calls | Invalid observations |
|---|---|---|---:|---:|---:|---:|
| Lang-1 | prior | completed | 2 | 0 | 0 | 0 |
| Lang-1 | fixed_l1 | incomplete | 2 | 11235 | 12 | 2 |
| Lang-1 | highest_suspicion | incomplete | 2 | 13490 | 13 | 2 |
| Lang-1 | v2_utility | incomplete | 2 | 11743 | 13 | 2 |
| Lang-1 | full | incomplete | 2 | 16140 | 22 | 0 |
| Lang-1 | l1_graph | incomplete | 22 | 13455 | 14 | 4 |
| Lang-1 | graft_adaptive | incomplete | 2 | 13352 | 15 | 2 |
| Math-1 | prior | failed | None | None | 0 | 0 |
| Math-1 | fixed_l1 | failed | None | None | 0 | 0 |
| Math-1 | highest_suspicion | failed | None | None | 0 | 0 |
| Math-1 | v2_utility | failed | None | None | 0 | 0 |
| Math-1 | full | failed | None | None | 0 | 0 |
| Math-1 | l1_graph | failed | None | None | 0 | 0 |
| Math-1 | graft_adaptive | failed | None | None | 0 | 0 |
| Time-1 | prior | completed | 1 | 0 | 0 | 0 |
| Time-1 | fixed_l1 | incomplete | 1 | 12902 | 12 | 2 |
| Time-1 | highest_suspicion | incomplete | 1 | 13326 | 13 | 2 |
| Time-1 | v2_utility | incomplete | 1 | 13301 | 13 | 2 |
| Time-1 | full | incomplete | 1 | 15753 | 23 | 0 |
| Time-1 | l1_graph | incomplete | 1 | 15000 | 14 | 4 |
| Time-1 | graft_adaptive | incomplete | 1 | 13900 | 13 | 3 |

Ranks from incomplete states are partial observations and must not be compared as completed accuracy. Missing labels are unavailable, not zero accuracy. The run REPORT.md separates operational and matched cohorts.

## Accounting and reproducibility

```json
{
  "known_provider_tokens": 62893,
  "reserved_or_unknown_tokens": 0,
  "new_calls": 76,
  "usage_unknown": false,
  "pending": false,
  "physical_inference_tokens": 62893,
  "simulated_provider_tokens": 0,
  "physical_token_definition": "Known provider input+output counters; unknown reservations separately retained",
  "synthetic": false
}
```

All reconstructed final rankings and logical paid-call sums matched saved arms. Journal physical counters match accounting.json. No recorded logical-budget violation occurred. Exact requests/responses and selected/reused fingerprints remain in the source run. Logical policy costs include reused observations; physical totals count each paid request once. Retrieval/embedding/build costs are excluded.

## Upgrade traces (offline interpretation)

| Bug | Policy | Stage | Selected candidate | Rank before | Rank after | MRR change | Tokens |
|---|---|---|---|---:|---:|---:|---:|
| Lang-1 | highest_suspicion | method_full | org.apache.commons.lang3.math.NumberUtils.createNumber(String) | 2 | 2 | 0.0000 | 2255 |
| Lang-1 | v2_utility | method_full | org.apache.commons.lang3.math.NumberUtils.createLong(String) | 2 | 2 | 0.0000 | 508 |
| Lang-1 | l1_graph | graph | org.apache.commons.lang3.math.NumberUtils.createInteger(String) | 2 | 1 | 0.5000 | 1110 |
| Lang-1 | l1_graph | graph | org.apache.commons.lang3.math.NumberUtils.createNumber(String) | 1 | 22 | -0.9545 | 1110 |
| Lang-1 | graft_adaptive | graph | org.apache.commons.lang3.StringUtils.isBlank(CharSequence) | 2 | 2 | 0.0000 | 775 |
| Lang-1 | graft_adaptive | method_full | org.apache.commons.lang3.StringUtils.isBlank(CharSequence) | 2 | 2 | 0.0000 | 570 |
| Lang-1 | graft_adaptive | graph | org.apache.commons.lang3.StringUtils.isEmpty(CharSequence) | 2 | 2 | 0.0000 | 772 |
| Time-1 | highest_suspicion | method_full | org.joda.time.field.UnsupportedDurationField.compareTo(DurationField) | 1 | 1 | 0.0000 | 424 |
| Time-1 | v2_utility | method_full | org.joda.time.Partial.size() | 1 | 1 | 0.0000 | 399 |
| Time-1 | l1_graph | graph | org.joda.time.field.UnsupportedDurationField.compareTo(DurationField) | 1 | 1 | 0.0000 | 978 |
| Time-1 | l1_graph | graph | org.joda.time.chrono.BuddhistChronology.assemble(Fields) | 1 | 1 | 0.0000 | 1120 |
| Time-1 | graft_adaptive | graph | org.joda.time.chrono.AssembledChronology.Fields.copyFieldsFrom(Chronology) | 1 | 1 | 0.0000 | 998 |

These marginal changes use gold labels only after execution. They are descriptive outcomes of these states, not deployable utility features. More graph evidence need not improve ranking, and method Full is not an oracle. Two reused development defects cannot establish statistical superiority or justify training a calibrated utility model.

Sources: manifest.json, environment.json, results.json, per-instance input/evaluation_only files and journal.sqlite3 in the recorded run. audit.json retains the provider/model identity and metric rows; ranking_evolution.json retains every paid transition.
