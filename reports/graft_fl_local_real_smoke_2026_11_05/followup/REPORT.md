# GRAFT local real-model smoke results

Fresh local Llama3.1 8B development smoke; this is not an A100 measurement or independent held-out validation. Existing Time-1/Lang-1/Math-1 inputs were selected for availability before inference; the separate Lang-54/Math-80/Closure-61 protocol remains unexecuted.

The evidence is supplied trigger-test text, not newly generated observed failure logs. Official benchmark tests were not run. Math-1 source-span incompatibility remains documented in the initial run. Condition: local_real_smoke_followup_20261105. All arms share the pool and configured fusion. Context: 8192; output per candidate: 160; batch size: 1; logical verification budget: 32000. Physical caps: 60000 run / 30000 per bug / 100 new requests.

| Bug | Policy | Status | Strict gold rank | Verification tokens | Calls | Invalid observations |
|---|---|---|---:|---:|---:|---:|
| Lang-1 | prior | completed | 2 | 0 | 0 | 0 |
| Lang-1 | fixed_l1 | completed | 2 | 15901 | 24 | 0 |
| Lang-1 | highest_suspicion | completed | 2 | 16414 | 25 | 0 |
| Lang-1 | v2_utility | completed | 2 | 15901 | 24 | 0 |
| Lang-1 | full | completed | 2 | 17477 | 24 | 0 |
| Lang-1 | l1_graph | completed | 2 | 17905 | 26 | 0 |
| Lang-1 | graft_adaptive | completed | 2 | 17849 | 27 | 0 |
| Time-1 | prior | completed | 1 | 0 | 0 | 0 |
| Time-1 | fixed_l1 | completed | 1 | 17730 | 24 | 0 |
| Time-1 | highest_suspicion | completed | 2 | 18154 | 25 | 0 |
| Time-1 | v2_utility | completed | 1 | 17730 | 24 | 0 |
| Time-1 | full | completed | 1 | 16133 | 24 | 0 |
| Time-1 | l1_graph | completed | 1 | 19626 | 26 | 0 |
| Time-1 | graft_adaptive | completed | 1 | 19667 | 27 | 0 |

Ranks from incomplete states are partial observations and must not be compared as completed accuracy. Missing labels are unavailable, not zero accuracy. The run REPORT.md separates operational and matched cohorts.

## Accounting and reproducibility

```json
{
  "known_provider_tokens": 42201,
  "reserved_or_unknown_tokens": 0,
  "new_calls": 59,
  "usage_unknown": false,
  "pending": false,
  "physical_inference_tokens": 42201,
  "simulated_provider_tokens": 0,
  "physical_token_definition": "Known provider input+output counters; unknown reservations separately retained",
  "synthetic": false
}
```

All reconstructed final rankings and logical paid-call sums matched saved arms. Journal physical counters match accounting.json. No recorded logical-budget violation occurred. Exact requests/responses and selected/reused fingerprints remain in the source run. Logical policy costs include reused observations; physical totals count each paid request once. Retrieval/embedding/build costs are excluded.

## Upgrade traces (offline interpretation)

| Bug | Policy | Stage | Selected candidate | Rank before | Rank after | MRR change | Tokens |
|---|---|---|---|---:|---:|---:|---:|
| Lang-1 | highest_suspicion | method_full | org.apache.commons.lang3.math.NumberUtils.createInteger(String) | 2 | 2 | 0.0000 | 513 |
| Lang-1 | l1_graph | graph | org.apache.commons.lang3.math.NumberUtils.createInteger(String) | 2 | 2 | 0.0000 | 1003 |
| Lang-1 | l1_graph | graph | org.apache.commons.lang3.math.NumberUtils.createNumber(String) | 2 | 2 | 0.0000 | 1001 |
| Lang-1 | graft_adaptive | graph | org.apache.commons.lang3.StringUtils.isEmpty(CharSequence) | 2 | 2 | 0.0000 | 704 |
| Lang-1 | graft_adaptive | method_full | org.apache.commons.lang3.StringUtils.isEmpty(CharSequence) | 2 | 2 | 0.0000 | 537 |
| Lang-1 | graft_adaptive | graph | org.apache.commons.lang3.StringUtils.isBlank(CharSequence) | 2 | 2 | 0.0000 | 707 |
| Time-1 | highest_suspicion | method_full | org.joda.time.field.UnsupportedDurationField.compareTo(DurationField) | 1 | 2 | -0.5000 | 424 |
| Time-1 | l1_graph | graph | org.joda.time.field.UnsupportedDurationField.compareTo(DurationField) | 1 | 1 | 0.0000 | 942 |
| Time-1 | l1_graph | graph | org.joda.time.chrono.BuddhistChronology.assemble(Fields) | 1 | 1 | 0.0000 | 954 |
| Time-1 | graft_adaptive | graph | org.joda.time.format.DateTimeFormatterBuilder.printTo(Writer,ReadablePartial,Locale) | 1 | 1 | 0.0000 | 775 |
| Time-1 | graft_adaptive | method_full | org.joda.time.format.DateTimeFormatterBuilder.printTo(Writer,ReadablePartial,Locale) | 1 | 1 | 0.0000 | 395 |
| Time-1 | graft_adaptive | graph | org.joda.time.format.DateTimeFormatterBuilder.printTo(StringBuffer,ReadablePartial,Locale) | 1 | 1 | 0.0000 | 767 |

These marginal changes use gold labels only after execution. They are descriptive outcomes of these states, not deployable utility features. More graph evidence need not improve ranking, and method Full is not an oracle. Two reused development defects cannot establish statistical superiority or justify training a calibrated utility model.

Sources: manifest.json, environment.json, results.json, per-instance input/evaluation_only files and journal.sqlite3 in the recorded run. audit.json retains the provider/model identity and metric rows; ranking_evolution.json retains every paid transition.
