# GRAFT execution report

**Development experiment; held-out status must be established from a preselected split manifest.**

Selected instances: 1. Arm statuses: {'completed': 12}.
Verification cost is provider input plus output, including paid invalid responses. Retrieval, embedding, test execution and whole-pipeline cost are separate. Exact-cache reuse retains logical policy cost but creates no new provider request.

| Profile | Cohort | Policy | Fusion | Labeled n | Completed | Top-1 | Top-5 | MRR | Mean verification tokens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| legacy_v1 | known_labeled_operational | fixed_l1 | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| legacy_v1 | matched_completed_labeled | fixed_l1 | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| legacy_v1 | known_labeled_operational | fixed_l1 | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| legacy_v1 | matched_completed_labeled | fixed_l1 | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| legacy_v1 | known_labeled_operational | l1_anchor | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| legacy_v1 | matched_completed_labeled | l1_anchor | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| legacy_v1 | known_labeled_operational | l1_anchor | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| legacy_v1 | matched_completed_labeled | l1_anchor | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| legacy_v1 | known_labeled_operational | l1_graph_focused | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11469.0 |
| legacy_v1 | matched_completed_labeled | l1_graph_focused | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11469.0 |
| legacy_v1 | known_labeled_operational | l1_graph_focused | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11469.0 |
| legacy_v1 | matched_completed_labeled | l1_graph_focused | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11469.0 |
| legacy_v1 | known_labeled_operational | l1_graph_test_aware | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| legacy_v1 | matched_completed_labeled | l1_graph_test_aware | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| legacy_v1 | known_labeled_operational | l1_graph_test_aware | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| legacy_v1 | matched_completed_labeled | l1_graph_test_aware | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| legacy_v1 | known_labeled_operational | l1_graph_test_aware_nostop | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| legacy_v1 | matched_completed_labeled | l1_graph_test_aware_nostop | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| legacy_v1 | known_labeled_operational | l1_graph_test_aware_nostop | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| legacy_v1 | matched_completed_labeled | l1_graph_test_aware_nostop | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| legacy_v1 | known_labeled_operational | prior | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |
| legacy_v1 | matched_completed_labeled | prior | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |
| legacy_v1 | known_labeled_operational | prior | suspicion_rank_blend | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |
| legacy_v1 | matched_completed_labeled | prior | suspicion_rank_blend | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |
| java_strict_v2 | known_labeled_operational | fixed_l1 | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| java_strict_v2 | matched_completed_labeled | fixed_l1 | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| java_strict_v2 | known_labeled_operational | fixed_l1 | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| java_strict_v2 | matched_completed_labeled | fixed_l1 | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| java_strict_v2 | known_labeled_operational | l1_anchor | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| java_strict_v2 | matched_completed_labeled | l1_anchor | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 10150.0 |
| java_strict_v2 | known_labeled_operational | l1_anchor | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| java_strict_v2 | matched_completed_labeled | l1_anchor | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 10150.0 |
| java_strict_v2 | known_labeled_operational | l1_graph_focused | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11469.0 |
| java_strict_v2 | matched_completed_labeled | l1_graph_focused | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11469.0 |
| java_strict_v2 | known_labeled_operational | l1_graph_focused | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11469.0 |
| java_strict_v2 | matched_completed_labeled | l1_graph_focused | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11469.0 |
| java_strict_v2 | known_labeled_operational | l1_graph_test_aware | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| java_strict_v2 | matched_completed_labeled | l1_graph_test_aware | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| java_strict_v2 | known_labeled_operational | l1_graph_test_aware | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| java_strict_v2 | matched_completed_labeled | l1_graph_test_aware | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| java_strict_v2 | known_labeled_operational | l1_graph_test_aware_nostop | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| java_strict_v2 | matched_completed_labeled | l1_graph_test_aware_nostop | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1667 | 11487.0 |
| java_strict_v2 | known_labeled_operational | l1_graph_test_aware_nostop | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| java_strict_v2 | matched_completed_labeled | l1_graph_test_aware_nostop | suspicion_rank_blend | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 11487.0 |
| java_strict_v2 | known_labeled_operational | prior | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |
| java_strict_v2 | matched_completed_labeled | prior | rrf | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |
| java_strict_v2 | known_labeled_operational | prior | suspicion_rank_blend | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |
| java_strict_v2 | matched_completed_labeled | prior | suspicion_rank_blend | 1 | 1 | 0.0000 | 0.0000 | 0.1000 | 0.0 |

Operational scores count incomplete labeled arms as zero; matched scores require every configured comparison to complete with labels. Unlabeled instances are not zero-accuracy defects. Empty matched cohorts are omitted, not evidence of equivalence.

Actual spending and unresolved reservations:

```json
{
  "known_provider_tokens": 12806,
  "reserved_or_unknown_tokens": 0,
  "new_calls": 27,
  "usage_unknown": false,
  "pending": false,
  "physical_inference_tokens": 12806,
  "simulated_provider_tokens": 0,
  "physical_token_definition": "Known provider input+output counters; unknown reservations separately retained",
  "synthetic": false
}
```

Source: results.json, accounting.json and manifest.json in this run. Per-arm files retain selected/rejected actions, paid observations and stage histories. raw_ledgers/requests.jsonl and journal.sqlite3 preserve exact provider requests and counters. No cache-based wall-time speedup is claimed.
