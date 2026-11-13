# Lang-54 manual smoke checkpoint

The manual run uses the existing `GRAFT-FL` implementation with the Lang-only certificate and the live-attested Lang-only cohort. The checkpoint preserves the accepted request payloads and selection records. This review made zero generation requests.

## Verified condition

| Item | Verified value |
|---|---|
| Buggy revision | `77a3c20bc0f30ca44f30aadad7f8ef0bc2528d0b` |
| Comparison | T0 propagation disabled, T1 propagation enabled |
| Schedule | 15 fixed candidates per arm, at most 30 generation requests |
| Prompt and output caps | 1,024 complete serving tokens and 160 generated tokens |
| Ranking | RRF, `k=60`, original prior order for ties |
| Pending hold | 8,352 tokens per request |
| Arm limits | 36,000 logical and 36,000 physical tokens, 24 calls |
| Defect limits | 72,000 physical tokens, 48 calls |
| Configured run limits | 216,000 physical tokens, 144 calls, with only 30 certified requests in the Lang-only schedule |
| T0 / T1 serial peaks | 24,819 / 24,787 tokens |
| Lang pair serial peak / settled upper bound | 42,438 / 35,257 tokens |

Both arms' exact payloads, evidence selections, source graph, pool, schedule and BGE-M3 bundle identities matched the saved native-tokenizer plan. The model is `llama3.1:8b` Q4_K_M on the audited Ollama 0.17.7 compatibility runner. The active endpoint's metadata and process identity matched the existing sealed lock. The Lang certificate and cohort claim were unused at this checkpoint.

## Local artifacts required to reproduce the lock

The source checkpoint includes code, tests, configuration and protocol documentation. Dataset material, embeddings, weights, native token captures, generated evidence comparisons, cohort locks and runtime state remain local. The configuration uses repository-relative artifact paths. Preserve the existing local artifacts to reuse the exact lock. Re-encoding may produce a different bundle identity even when the source and checkpoint match.

| Artifact identity | SHA-256 |
|---|---|
| Lang certificate file | `477966eb636f45fef2fe754522032b5c569f9e2f26fbbd5c06ade9eb3bfed2a4` |
| Lang cohort canonical checksum | `4e8fb8bee7e630a7ddec1a82455d74b1813a08384f4f5307234f0a4f9203e435` |
| Online manifest bytes | `992e93363e53a24cff1af0c0c73eb56b9037304c2bf31b38532678997e15a113` |
| Offline label manifest bytes | `68a908d16dbe0508777e033ed9e56d8b447db50eae51ede279b90f95a1185ef6` |
| Lang embedding bundle bytes | `c0a2babe9e0276d84b496299464014d517e0c6daf469d0f9dfacc5e934cc2e80` |
| Installed Llama3.1 manifest | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` |
| Installed no-tools Go template | `948af2743fc78a328dcb3b0f5a31b3d75f415840fdb699e8b1235978392ecf85` |

The [Mac live collector runbook](MAC_LIVE_COLLECTOR_READINESS.md) records how to regenerate the Lang lock with `preflight --attest-live` if Ollama restarts before collection. The source checkpoint keeps the previously accepted implementation byte-for-byte so its Git commit does not invalidate the existing cohort. The readiness test history and known workspace baseline failures are recorded in the runbook.

## After manual collection

Use `manual-real-lang54` as the output directory under `results/graft_fl_real_preparation/`. Start the review with `manifest.json`, `accounting.json`, `journal.sqlite3`, and `raw_ledgers/requests.jsonl`. For a sealed result, the portable `report` command reconciles the journal and regenerates tables without inference. Verify that the paid rows match the frozen schedule, then report T0/T1 completion, invalid responses, actual provider input and output tokens, reservation and drift events, ranking validity, and Top-1, Top-5 and MRR. If a run stops with unresolved usage, preserve its artifacts for review before resuming. Math-80, Closure-61 and Linux/A100 execution require separate approval.
