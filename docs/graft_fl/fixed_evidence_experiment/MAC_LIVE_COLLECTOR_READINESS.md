# Mac live collector readiness

13 November 2026 · `GRAFT-FL`

**Mac engineering gate PASS. First real-model request pending separate approval.** The existing fixed-evidence runner and portable launcher now support explicit manual collection for a live-attested Ollama cohort. The readiness checks queried metadata on the already-running Mac endpoint. They made zero generation requests and started no server, runner job or download. A loopback fake provider exercised the production transport, collector, journal, resume and reporting paths.

## Serving provenance and execution contract

The launcher requires a direct IPv4 loopback listener under the current account. It inspects the owning process, verifies its executable against the audited binary, and reads the serving arguments and initial environment to establish compatibility mode and the model store. The active endpoint reported Ollama 0.17.7. The model manifest, GGUF, defaults, config and installed Go template matched their expected digests. The active `/api/show` metadata also matched the template and local GGUF path. This validation uses `/api/version`, `/api/tags` and the metadata-only `/api/show` endpoint. The compatibility decision follows the versioned Ollama implementation for the text-only `llama` architecture. [Ollama compatibility runner selection](https://github.com/ollama/ollama/blob/v0.17.7/llm/server.go), [Ollama runner environment](https://github.com/ollama/ollama/blob/v0.17.7/envconfig/config.go).

`--attest-live` seals that provenance, the endpoint and process identity, the selected schedule, all prepared request fingerprints and the budget certificate into the cohort. An ordinary offline plan cannot authorize live dispatch. `run` and `resume` require `--allow-inference`, the sealed live cohort and the same certified schedule. Before each fresh request the transport rechecks the serving metadata, listener and model-file state. The direct connection disables proxy and redirect behavior. If Ollama restarts before a first run, regenerate a fresh live preflight for the same certificate. If it restarts during a paid run, stop for explicit review of the journal and serving provenance. A cohort or output directory change cannot silently restart a claimed certificate. Keep the run directory and its claim in the model store's sibling `graft-fl-live-claims` directory.

Every request keeps its 8,352-token hold until completed usage is known and matches the certificate. A completed invalid verdict remains charged and its arm is incomplete. A response with a different model identity, prompt-count drift or excessive output retains paid counters and the hold and blocks the run. A disconnect or missing usage leaves the reservation unresolved. The journal reuses confirmed paid responses when resuming. The report reconstructs actual provider counters and displays planning bounds separately. Generation accuracy and runtime behavior on the real model remain unmeasured. Keep the Ollama installation and model tag stable during collection. The checks detect ordinary process and file drift and cannot provide atomic protection against privileged concurrent mutation of the host.

## Frozen cohorts

Both cohorts preserve the same T0/T1 prompt bytes, schedules, embeddings, source graphs, 1,024 complete serving-token allowance, 160-token output cap, generation settings and RRF fusion. The sole arm difference is graph propagation. Closure-61 is excluded. Its source-coordinate discrepancy, frozen rankings and candidate-recall miss are unchanged.

| Scope | Planned calls | Settled upper bound | Peak reservation | Applicable physical cap | Status |
|---|---:|---:|---:|---:|---|
| Lang-54 T0 | 15 | 17,651 | 24,819 | 36,000 | PASS |
| Lang-54 T1 | 15 | 17,606 | 24,787 | 36,000 | PASS |
| Lang-54 pair | 30 | 35,257 | 42,438 | 72,000 | PASS |
| Math-80 T0 | 21 | 24,669 | 31,846 | 36,000 | PASS |
| Math-80 T1 | 21 | 24,680 | 31,860 | 36,000 | PASS |
| Math-80 pair | 42 | 49,349 | 56,529 | 72,000 | PASS |
| Lang-only cohort | 30 | 35,257 | 42,438 | 216,000 | PASS |
| Lang and Math cohort | 72 | 84,606 | 91,786 | 216,000 | PASS |

The sealed plans are in `results/graft_fl_real_preparation/ollama0177-tokenizer-verified/live-ready-lang54/` and `live-ready-lang54-math80/`. The Lang-only certificate was generated from the 30 Lang request IDs in the native audit. The two-defect certificate contains 72 request IDs. Their schedule, certificate digest and cohort lock are independently checked. If both experiments are approved and run, Lang is billed separately in each independent journal. Its two executions are repetitions of one defect.

## Test results

The focused suite passed **56 tests**. The GRAFT/RAVE regressions passed **298 tests**. The full workspace suite returned **319 passed and 3 baseline failures**. The failures are the same Agent4SR format-only finalization test and two dependency tests for the absent `scripts/run_pipeline.sh` recorded in the [settlement amendment](PROMPT_SETTLEMENT_AMENDMENT.md). The two regression runs restricted network connections to loopback servers created by the tests.

```bash
python -B -m pytest -q tests/test_graft_ollama_live.py \
  tests/test_graft_prompt_settlement.py tests/test_graft_fixed_evidence_schedule.py \
  tests/test_graft_phase4_reporting.py tests/test_graft_phase5_portable.py
```

## Exact commands for manual use

Run from the repository root with the project Python environment. These variables select the audited local endpoint and resources. Use the already sealed cohorts below. To recreate one, run the corresponding metadata-only preflight with a fresh output directory, then update that cohort path in the run command.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
graft_audit=results/graft_fl_real_preparation/ollama0177-tokenizer-verified
graft_ollama="$HOME/.ollama/models"
graft_binary=/Applications/Ollama.app/Contents/Resources/ollama
graft_manifest="$graft_ollama/manifests/registry.ollama.ai/library/llama3.1/8b"
graft_lang_config=configs/graft_portable/real_mac_prompt_settlement_lang54.json
graft_pair_config=configs/graft_portable/real_mac_prompt_settlement.json
```

Metadata-only regeneration for Lang-54 and the two-defect cohort respectively.

```bash
python -B -m rave_fl.graft.portable_experiment preflight --attest-live --profile local_mac \
  --config "$graft_lang_config" --serving-binary "$graft_binary" \
  --generation-manifest "$graft_manifest" --ollama-models-dir "$graft_ollama" \
  --output "$graft_audit/live-ready-lang54-repeat"

python -B -m rave_fl.graft.portable_experiment preflight --attest-live --profile local_mac \
  --config "$graft_pair_config" --serving-binary "$graft_binary" \
  --generation-manifest "$graft_manifest" --ollama-models-dir "$graft_ollama" \
  --output "$graft_audit/live-ready-lang54-math80-repeat"
```

After separate approval, the Lang-54-only smoke makes at most 30 generation requests. It uses the Lang-only certificate and cohort.

```bash
graft_lang_run=results/graft_fl_real_preparation/manual-real-lang54
python -B -m rave_fl.graft.portable_experiment run --profile local_mac --allow-inference \
  --config "$graft_lang_config" --cohort "$graft_audit/live-ready-lang54/cohort.json" \
  --serving-binary "$graft_binary" --generation-manifest "$graft_manifest" \
  --ollama-models-dir "$graft_ollama" --output "$graft_lang_run"
```

The independently approved full Lang-54 and Math-80 experiment makes at most 72 generation requests.

```bash
graft_pair_run=results/graft_fl_real_preparation/manual-real-lang54-math80
python -B -m rave_fl.graft.portable_experiment run --profile local_mac --allow-inference \
  --config "$graft_pair_config" --cohort "$graft_audit/live-ready-lang54-math80/cohort.json" \
  --serving-binary "$graft_binary" --generation-manifest "$graft_manifest" \
  --ollama-models-dir "$graft_ollama" --output "$graft_pair_run"
```

For a resume, substitute `resume` for `run` and use the same cohort, resource paths and output directory. Exit 3 with a sealed result indicates an incomplete pair, including a paid invalid verdict. An execution error with uncertain usage requires journal review before further calls.

```bash
python -B -m rave_fl.graft.portable_experiment report \
  --run "$graft_lang_run" --output "${graft_lang_run}-report"
```

**Linux/A100 remains BLOCKED pending host validation.** The same implementation inspects a local Linux listener through `lsof` and `/proc`. The Mac certificate cannot authorize Linux. A server-specific audit must validate the serving binary, runner, model, template and prompt token IDs before sealing a Linux cohort. No CUDA or server job was run in this gate.
