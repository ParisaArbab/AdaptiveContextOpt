# GRAFT execution report

**Development experiment; held-out status must be established from a preselected split manifest.**

Selected instances: 3. Arm statuses: {'completed': 15}.
Verification cost is provider input plus output, including paid invalid responses. Retrieval, embedding, test execution and whole-pipeline cost are separate. Exact-cache reuse retains logical policy cost but creates no new provider request.

| Profile | Cohort | Policy | Fusion | Labeled n | Completed | Top-1 | Top-5 | MRR | Mean verification tokens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| legacy_v1 | known_labeled_operational | fixed_l1 | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 15695.0 |
| legacy_v1 | matched_completed_labeled | fixed_l1 | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 15695.0 |
| legacy_v1 | known_labeled_operational | graft_adaptive | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17953.0 |
| legacy_v1 | matched_completed_labeled | graft_adaptive | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17953.0 |
| legacy_v1 | known_labeled_operational | highest_suspicion | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 16725.7 |
| legacy_v1 | matched_completed_labeled | highest_suspicion | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 16725.7 |
| legacy_v1 | known_labeled_operational | l1_graph | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17745.7 |
| legacy_v1 | matched_completed_labeled | l1_graph | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17745.7 |
| legacy_v1 | known_labeled_operational | prior | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 0.0 |
| legacy_v1 | matched_completed_labeled | prior | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 0.0 |
| java_strict_v2 | known_labeled_operational | fixed_l1 | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 15695.0 |
| java_strict_v2 | matched_completed_labeled | fixed_l1 | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 15695.0 |
| java_strict_v2 | known_labeled_operational | graft_adaptive | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17953.0 |
| java_strict_v2 | matched_completed_labeled | graft_adaptive | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17953.0 |
| java_strict_v2 | known_labeled_operational | highest_suspicion | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 16725.7 |
| java_strict_v2 | matched_completed_labeled | highest_suspicion | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 16725.7 |
| java_strict_v2 | known_labeled_operational | l1_graph | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17745.7 |
| java_strict_v2 | matched_completed_labeled | l1_graph | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 17745.7 |
| java_strict_v2 | known_labeled_operational | prior | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 0.0 |
| java_strict_v2 | matched_completed_labeled | prior | rrf | 3 | 3 | 0.6667 | 1.0000 | 0.8333 | 0.0 |

Operational scores count incomplete labeled arms as zero; matched scores require every configured comparison to complete with labels. Unlabeled instances are not zero-accuracy defects. Empty matched cohorts are omitted, not evidence of equivalence.

Actual spending and unresolved reservations:

```json
{
  "known_provider_tokens": 62141,
  "reserved_or_unknown_tokens": 0,
  "new_calls": 84,
  "usage_unknown": false,
  "pending": false,
  "physical_inference_tokens": 62141,
  "simulated_provider_tokens": 0,
  "physical_token_definition": "Known provider input+output counters; unknown reservations separately retained",
  "synthetic": false
}
```

Source: results.json, accounting.json and manifest.json in this run. Per-arm files retain selected/rejected actions, paid observations and stage histories. raw_ledgers/requests.jsonl and journal.sqlite3 preserve exact provider requests and counters. No cache-based wall-time speedup is claimed.
