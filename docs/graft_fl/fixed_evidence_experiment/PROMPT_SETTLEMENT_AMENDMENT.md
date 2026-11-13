# Prompt-aware settlement for the fixed T0/T1 smoke

The [Mac live collector readiness report](MAC_LIVE_COLLECTOR_READINESS.md) records the subsequent metadata attestation and exact manual commands. The status below describes the earlier offline-only gate.

13 November 2026 · Branch `GRAFT-FL`

The approved execution-admission amendment keeps a full **8,352-token hold** while each call is pending. Once a completed response provides valid usage counters, admission charges a settled upper bound of the verified complete prompt length plus the **160-token output cap**. The ledger records provider-reported input and output separately. A completed response with an invalid verdict remains paid and makes its arm incomplete. If token usage drifts from the verified input count, the journal records the paid usage, retains the hold and blocks further execution. A transport error, incomplete response or interrupted dispatch keeps the full hold. Known counters from an incomplete response remain visible alongside its unresolved hold.

The original preregistration and default configuration are preserved. The separate [smoke configuration](../../../configs/graft_portable/real_mac_prompt_settlement.json) applies the amendment to the full Lang-54 and Math-80 schedules. It retains the 1,024-token complete prompt allowance, source graph, candidate pools and schedules, pinned real BGE-M3 bundles, prompt bytes, model options, output cap and RRF fusion. Propagation is the sole difference between T0 and T1. Closure-61 remains outside this smoke and its candidate-recall miss is preserved.

## Proof and accounting

- The [certificate builder](../../../scripts/build_graft_settlement_certificate.py) reads the archived native Ollama token capture and extracts an entry for every scheduled action, binding the exact JSON payload fingerprint and token-ID fingerprint to the model, Go template, serving version, binary, runner mode, dataset, graph, pool, schedule and embedding hashes. The smoke configuration pins the certificate by SHA-256. The planner recomputes each token sequence against the verified tokenizer files. Portable preflight also verifies the local serving binary, host, manifest, model blob and template bytes.
- For a serial schedule, the planned exposure at request `i` is `8,352 + sum(prompt_tokens[j] + 160 for j < i)`. The same admission rule applies at the arm, defect and cohort boundaries. Historical hold sums remain visible as a diagnostic. They do not represent peak exposure or usage.
- The store persists the full hold and request proof together in SQLite before dispatch, and the raw ledger records the same proof. A complete response persists its provider counters before the hold is released. Backend and report checks reconcile the bound, actual counters and expected request key. A crash after the response transaction and before the raw response event may recover that final event from the durable journal. Pending, uncertain or violating requests block resume. Resume can reuse complete journal entries without paying again.

## Mac offline readiness

All 72 requests have verified serving-token counts and reproduce the earlier audited request payloads. All four arms and both pairs pass the original caps. Provider actual usage is unavailable because the dry-run made zero generation requests.

| Defect | Arm | Calls | Sum of exact prompts | Settled upper bound | Serial peak with pending hold | Arm cap | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Lang-54 | T0 | 15 | 15,251 | 17,651 | 24,819 | 36,000 | PASS |
| Lang-54 | T1 | 15 | 15,206 | 17,606 | 24,787 | 36,000 | PASS |
| Math-80 | T0 | 21 | 21,309 | 24,669 | 31,846 | 36,000 | PASS |
| Math-80 | T1 | 21 | 21,320 | 24,680 | 31,860 | 36,000 | PASS |

| Scope | Calls | Settled upper bound | Serial peak | Physical cap | Status |
|---|---:|---:|---:|---:|---|
| Lang-54 | 30 | 35,257 | 42,438 | 72,000 | PASS |
| Math-80 | 42 | 49,349 | 56,529 | 72,000 | PASS |
| Cohort | 72 | 84,606 | 91,786 | 216,000 | PASS |

The fixture coverage includes exact and one-below cap boundaries, invalid paid verdicts, prompt-ID and provider count drift, output overflow, interrupted and partial requests, durable journal reconciliation, sealed reporting and resume. The live collector remains disabled in both entry points. Linux/A100 must produce and validate its own serving-binary and runner-specific tokenizer certificate. The Mac certificate fails closed on a different host or binary. The current certificate covers this two-defect serial schedule.

The focused suite passed **52 tests**. The GRAFT/RAVE regression suite passed **280 tests**. The full workspace suite reported **301 passed and 3 unrelated failures**. `test_format_only_finalization_uses_seen_entities` also fails on an isolated archive of the unchanged branch HEAD. The two existing `test_pipeline_dependencies.py` tests expect `scripts/run_pipeline.sh`, which is absent from this checkout and branch HEAD. The regression runs allowed only local test-fixture connections.

## Reproduce the offline preflight

Use the project Python environment and the previously provisioned Mac artifacts. Choose a fresh output directory. The pinned certificate and config can also be regenerated from the existing native capture using the builder with fresh paths.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
graft_audit=results/graft_fl_real_preparation/ollama0177-tokenizer-verified
graft_config=configs/graft_portable/real_mac_prompt_settlement.json
graft_ollama="$HOME/.ollama/models"
graft_preflight="$graft_audit/settlement-manual-preflight"

python -B -m rave_fl.graft.portable_experiment dry-run \
  --profile local_mac --config "$graft_config" \
  --serving-binary /Applications/Ollama.app/Contents/Resources/ollama \
  --generation-manifest "$graft_ollama/manifests/registry.ollama.ai/library/llama3.1/8b" \
  --ollama-models-dir "$graft_ollama" --output "$graft_preflight"
```

After separate approval and live-collector validation, the intended first manual run uses the same two-defect cohort. Regenerate the cohort lock after any implementation changes. The future collector must attest the identity of the endpoint serving the requests, including the compatibility runner, binary, model digest and template, before dispatch. Passing the offline check establishes the provenance of local artifacts. The endpoint still needs its own check.

```bash
graft_run=results/graft_fl_real_preparation/manual-real-smoke-000
python -B -m rave_fl.graft.portable_experiment run \
  --profile local_mac --config "$graft_config" --cohort "$graft_preflight/cohort.json" \
  --shard-index 0 --allow-inference --output "$graft_run"
```

After an approved live execution, use the same invocation with `resume` in place of `run` and the same output directory. An uncertain reservation requires reconciliation before further calls. A completed invalid verdict remains in the ledger and is excluded from matched-complete accuracy.
