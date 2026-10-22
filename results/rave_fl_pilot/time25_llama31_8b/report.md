# RAVE-FL adaptive pilot

Status: complete. Model: llama3.1:8b. Seed: 0.

Verification-stage experiment; shared pilot fusion ranker; no claim of full SieveFL reproduction.

| Bug | Arm | Calls | Screened/full | Actual tokens | Top-1 | Top-5 | MRR | Stop |
|---|---|---:|---:|---:|---|---|---:|---|
| Time-25 | full | 24 | 24/24 | 21378 | False | True | 0.500 | policy_complete |
| Time-25 | rank_threshold | 14 | 14/14 | 12887 | False | False | 0.111 | policy_complete |
| Time-25 | fixed_batch | 6 | 24/0 | 11298 | False | True | 0.250 | fixed_screen_complete |
| Time-25 | confidence | 53 | 24/24 | 50015 | False | True | 0.500 | all_required_checks_complete |
| Time-25 | adaptive | 53 | 24/24 | 50015 | False | True | 0.500 | all_required_checks_complete |

## Limits

- Preflight token estimates are not a provider-enforced hard cap.
- Shared pilot fusion ranker is not the original SieveFL final ranker.
- Bootstrap risk is untrained; candidate harm calibration does not certify issue-level stopping.
- All per-arm model requests, including invalid replies and repeated context, are charged.
- Upstream retrieval tokens are not included; this is the verification-stage comparison.
- Ollama counters measure logical tokens; monetary API cost is not available.

Paired confidence intervals are omitted for one bug. No tiny pilot establishes noninferiority or novelty.
