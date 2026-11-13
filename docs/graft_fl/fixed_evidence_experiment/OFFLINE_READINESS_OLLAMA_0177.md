# Offline readiness for the Ollama 0.17.7 smoke test

The [approved prompt-settlement amendment](PROMPT_SETTLEMENT_AMENDMENT.md) records the subsequent implementation and successful 72-request offline admission check under the original caps.

13 November 2026 · Branch `GRAFT-FL` · Base `e64d6dee9e642adaf08e4485512e7060f3335403`

**Prompt packing passes for Lang-54 and Math-80 at the original 1,024-token allowance. Full-schedule admission and live execution remain blocked.** The offline audit prepared both arms and all 72 requests with zero generation calls. The model, candidate schedules, prompts, allowance, output limit, propagation settings, and fusion follow the frozen protocol. The original configuration and execution caps are intact.

## Serving-equivalent token IDs

The installed desktop Ollama binary is version 0.17.7 with SHA-256 `323fa5b44270a9cd59ec4923d2f85dfd59ff97de9e2243c591329d7465e98784`. Its startup log records that version and `OLLAMA_NEW_ENGINE:false`. The pinned model is `llama3.1:8b` Q4_K_M, manifest SHA-256 `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`. The GGUF, config, and Go template bytes matched their layer digests. The config identifies a text-only `llama` family with no custom renderer. The separate Homebrew CLI plays no part in the capture.

For one system message and one user message with tools absent, the installed Go template renders the system and user headers, unmodified message content with end-of-turn markers, and the assistant header followed by two line feeds. The compatibility runner tokenizes the complete text with special-token insertion and parsing enabled. The local mirror implements that restricted rendering and the pinned GGUF vocabulary and BPE merges. Its chat-template file is a transcription of the **installed Go branch** for this request shape. An initial mirror check caught a trailing line feed dropped by the local template engine. The mirror now preserves both line feeds. The GGUF-embedded Jinja template was unused. [Ollama prompt path](https://github.com/ollama/ollama/blob/v0.17.7/server/prompt.go), [Go template execution](https://github.com/ollama/ollama/blob/v0.17.7/template/template.go), [Compatibility runner selection](https://github.com/ollama/ollama/blob/v0.17.7/llm/server.go), [Runner tokenization](https://github.com/ollama/ollama/blob/v0.17.7/runner/llamarunner/runner.go).

The offline native reference used the tokenizer inside a temporary copy of the installed executable with unchanged ARM64 code bytes. The copy was ad hoc signed for local debugging after macOS rejected debugger attachment to the installed binary. LLDB invoked the vocabulary-only loader and native `llama_tokenize` from the CLI `help` process. The native log confirms tensors were skipped. Reading the pinned model blob to verify its SHA-256 was a separate file-integrity check. No generation context, endpoint request, or server job was used. **4,589 distinct rendered inputs matched exactly**, including all mandatory packages, intermediate packing trials and selected prompts. Saved token ID arrays and rendered-text hashes permit per-request verification.

This establishes offline serving equivalence for the audited text-only request shape and observed Mac compatibility runner. The audit has no live provider-reported usage. Revalidate the serving build, template and token IDs on Linux/A100, and repeat the validation after changes to the model, backend, request shape or evidence cohort. The production planner still leaves `serving_template_tokens` unset. The independent audit binds each rendered request to the native count.

| Defect | Targets per arm | Mandatory prompt, serving tokens | T0 packed prompt range / total | T1 packed prompt range / total |
|---|---:|---:|---:|---:|
| Lang-54 | 15 | 602–628 | 1,009–1,024 / 15,251 | 1,005–1,024 / 15,206 |
| Math-80 | 21 | 680–728 | 1,007–1,024 / 21,309 | 1,005–1,024 / 21,320 |

The allowance covers the **complete chat prompt**, including headers and BOS, separately from the 160-token output cap. Every prompt retained its mandatory evidence. Both arms use the same candidate schedule, real BGE-M3 bundle, graph, packer, counter, and allowance. Propagation changed the optional evidence set for all 36 target pairs. Mean selected-node Jaccard overlap was 0.261 for Lang and 0.289 for Math. This is a selection diagnostic. Ranking and generation outcomes are still unknown.

## Full-schedule budget options

The current planner and journal use an 8,352-token hold per fresh request (`8,192 + 160`). The 36,000-token logical and physical arm caps fail before the full schedules can be authorized. Four such holds fit under each arm cap when summed for offline admission. Candidate schedules require 15 and 21 calls per arm. All costs below are upper bounds, with no cache or response-length savings assumed.

| Option | Lang T0 / T1 | Lang pair | Math T0 / T1 | Math pair | Both pairs |
|---|---:|---:|---:|---:|---:|
| Current full-window holds | 125,280 / 125,280 | 250,560 | 175,392 / 175,392 | 350,784 | 601,344 |
| Exact prompt length + 160, reserve and settle | 17,651 / 17,606 | 35,257 | 24,669 / 24,680 | 49,349 | 84,606 |
| Verified 1,024 + 160 uniform hold | 17,760 / 17,760 | 35,520 | 24,864 / 24,864 | 49,728 | 85,248 |
| Exact settlement bound, keep full-window pending hold | 24,819 / 24,787 | 42,438 | 31,846 / 31,860 | 56,529 | 91,786 |

The last row uses the recorded order (Lang then Math, T0 then T1 within each defect), with an 8,352-token hold plus the sum of exact prompt-and-output bounds for the settled prefix. All three prompt-aware options fit the original arm, bug, and cohort budgets on the verified schedule. The 72 calls fit the configured 144-call cohort ceiling. The compatibility runner derives `PromptEvalCount` from the length of its tokenized inputs and enforces the configured prediction cap. This supports the proposed offline bounds given the pinned build and exact requests. Actual output and provider counters remain to be measured. [Ollama v0.17.7 compatibility runner](https://github.com/ollama/ollama/blob/v0.17.7/runner/llamarunner/runner.go).

**Recommended amendment for approval:** keep the existing full-window pending hold and use the verified prompt length plus output cap as the maximum settled cost. Bind each bound to the exact request, serving binary and runner mode, template, model digest, and tokenizer receipt. Apply it consistently in the planner, backend, journal, and reporting. Retain the full pending hold and stop on unknown usage. Preserve paid responses and stop on count drift. Reject requests that fail provenance checks before dispatch. Require synthetic tests of these paths and a ready two-defect dry-run before enabling live execution. This changes the admission contract while retaining the experimental condition and caps. Exact prompt-aware reservation is also feasible but would narrow the hold on a pending request. The verified uniform bound is a simpler, slightly looser alternative if the live contract can enforce the 1,024-token invariant.

**Execution-cap alternative for separate approval:** keep the current planner and reservation rule and set shared arm logical and physical caps to at least **175,392**, per-bug physical cap to **350,784**, and two-defect physical cap to **601,344**. These are the minimum shared caps under the present schedule. Use a separate condition record for the execution-cap amendment, retain 72 as the overall new-call limit for this smoke test, and preserve both full candidate schedules. Raising caps alone leaves the real-collector gate closed.

## Closure-61 source validity and recall

The clean upstream buggy commit is `bac71221414f7f42d31d3634c935c2ae1a940376`. The pinned SBIR rank-9 candidate `NodeUtil.nodeTypeMayHaveSideEffects(Node,AbstractCompiler)` has upstream AST span **1003–1024**, while the historical CSV declares **1005–1026**. A byte comparison with the separate Defects4J fixed-revision-plus-patch reconstruction reproduced the two added lines in `NodeUtil.java`, a comment and blank line above the candidate. The pinned framework's `checkout_vid` also builds the benchmark buggy tree by applying the source patch to the fixed version. All 27 entries in the original top-10-per-ranker union mapped to their AST methods in the reconstruction. The original online checkout and all CSV bytes are intact. The strict exact-upstream importer therefore still blocks Closure.

The independently verified faulty overload, `NodeUtil.functionCallHasSideEffects(Node,AbstractCompiler)`, ranks **35th in SBIR, 34th in Ochiai and 126th in BoostN**. Candidate recall is a miss even before the 24-candidate cap. The source discrepancy and recall miss are separate findings. Keep Closure out of the Lang/Math smoke. For a later Closure run, first approve an explicit source-provenance decision. One option is a gold-blind AST coordinate reconciliation on the exact upstream tree, preserving every candidate ID, score, and position and requiring a unique source match. Another is to approve the canonical Defects4J buggy reconstruction as a separate source identity with independently reproduced failure evidence on that tree. Both retain the recall miss.

## Reproduction and future manual smoke

The provider-free regression suite passed **267 tests in 66.90 seconds**. The vocabulary-reader test checks that retaining token strings is opt-in and that the reader stops before the tensor payload. The Mac launcher dry-run reproduced all 72 audited request payloads and paired diagnostics. It exited 3 because the original caps exclude both pairs. Its admitted-call count is zero, while the pair records retain the full schedules and their required holds. All readiness artifacts are under `results/graft_fl_real_preparation/ollama0177-tokenizer-verified/`, including `native-comparison.json`, `paired-plan.json`, `mandatory-counts.json`, `readiness-summary.json`, `closure-coordinate-recheck.json`, and `mac-dry-run/preflight.json`. Use the existing Python 3.11 environment. Reproducing the native capture needs the audited Mac binary, local Xcode tools and existing model files. Normal packing uses the same cross-platform launcher and saved tokenizer directory.

```bash
graft_audit=results/graft_fl_real_preparation/ollama0177-tokenizer-verified
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PYTHONPATH="$PWD" python -B "$graft_audit/reproduce_native_token_audit.py" \
  results/graft_fl_real_preparation/ollama0177-tokenizer-repeat

graft_data=data/graft_real_preparation/online/validated-two-defects.json
graft_cfg=configs/graft_portable/real_mac_preflight.json
graft_ollama="$HOME/.ollama/models"
graft_host=local_mac
graft_preflight=results/graft_fl_real_preparation/manual-smoke-preflight

python -B -m rave_fl.graft.portable_experiment dry-run \
  --profile "$graft_host" --config "$graft_cfg" --dataset-file "$graft_data" \
  --bugs Lang-54,Math-80 --tokenizer-dir "$graft_audit/tokenizer" --max-new-calls 72 \
  --generation-manifest "$graft_ollama/manifests/registry.ollama.ai/library/llama3.1/8b" \
  --ollama-models-dir "$graft_ollama" --output "$graft_preflight"
```

The original caps yield exit 3. For the recommended amendment, use the approved future config with the tested admission contract and require a ready preflight. For the cap alternative, pass `--budget 175392 --per-arm-physical-budget 175392 --per-bug-physical-budget 350784 --physical-budget 601344` to both the preflight and the future run, after approval. The current `run` and `resume` commands explicitly reject real backends. The command below is the intended manual invocation once the real collector is separately approved, implemented, and validated. For Linux/A100, set `graft_host=remote_linux_a100`, supply the local model and evidence paths, and repeat the server-specific gates before launch.

```bash
python -B -m rave_fl.graft.portable_experiment run \
  --profile "$graft_host" --config "$graft_cfg" --dataset-file "$graft_data" \
  --bugs Lang-54,Math-80 --tokenizer-dir "$graft_audit/tokenizer" --max-new-calls 72 \
  --cohort "$graft_preflight/cohort.json" --shard-index 0 --allow-inference \
  --output results/graft_fl_real_preparation/manual-smoke-shard-000
```
