# Real-data readiness after scoped provisioning

The [Ollama 0.17.7 offline readiness audit](OFFLINE_READINESS_OLLAMA_0177.md) supersedes the byte-proxy packing result for Lang-54 and Math-80. It confirms prompt fit and documents the remaining full-schedule admission blocker.

13 November 2026. Branch `GRAFT-FL`, base commit `e64d6dee9e642adaf08e4485512e7060f3335403`. The Mac diagnostics reached the frozen T0/T1 planner on authentic Lang and Math inputs. The three-defect real-model run remains **BLOCKED**. The checks made zero generation requests and used the same scientific configuration for both arms, with propagation as the only arm difference.

## Input gates

| Defect | Exact upstream Git checkout | Reproduced failure and independently checked method label | Frozen candidate pool | Real-data readiness |
|---|---|---|---|---|
| Lang-54 | PASS, `77a3c20bc0f30ca44f30aadad7f8ef0bc2528d0b` | PASS, `testLang328` and `LocaleUtils.toLocale(String)` | PASS, 15 verified spans, label present | BLOCKED by prompt and reservation admission |
| Math-80 | PASS, `c78a0e3b003ac36d57a3c895fc8240ef66546a56` | PASS, `testMathpbx02` and `EigenDecompositionImpl.flipIfWarranted(int,int)` | PASS, 21 verified spans, label present | BLOCKED by prompt and reservation admission |
| Closure-61 | PASS, `bac71221414f7f42d31d3634c935c2ae1a940376` | PASS, three pinned triggers and `NodeUtil.functionCallHasSideEffects(Node,AbstractCompiler)` | BLOCKED on the exact upstream tree by a pinned CSV coordinate mismatch | BLOCKED before embedding export |

The approved downloads staged the Defects4J tree at `8c16da8230843cdc918eaf4ddb449637f02b83c6`, six upstream commit objects with their required trees, and the nine ranking CSVs at FlexFL reference revision `744143aa3e9c222b111beca55f5e75eab6d16285`. All three upstream HEADs and working trees were checked. The CSV receipts record pinned Git blob IDs and SHA-256 checksums. The archived triggers and patch objects were checked against the framework commit. The existing reference clones and prior results remain separate from the new staging directory.

The official Defects4J CLI probe with Java 11 stops at its project-repository initialization check. Its pinned launcher also requires the initialized Major/Ant, test-generation and Gradle layout. To reproduce the failures within the approved resources, an isolated JUnit harness reused the fixed revisions' trigger tests and checked every production file against the exact upstream buggy checkout before and after execution. It used local Java 11, the JUnit JAR in the pinned framework, and the locally installed Gradle distribution's Ant JARs for Closure. Lang's legacy source requires ISO-8859-1 and exclusion of the pre-Java-5 `enum` package. Lang and Math matched the archived exception and project frames. Closure matched the archived failure text and project frame method names for `testCall1`, `testCall2`, and `testRemoveUselessOps`. Closure's test-class line numbers were unavailable in this compilation. The reproduced failure logs, command receipts, and compile attempts are retained under `data/graft_real_preparation/offline/`.

Gold was derived from the exact buggy-to-fixed production diff and cross-checked against the pinned FlexFL labels. The evaluation labels and reference patches are outside the online input tree. The online data contains observed failure text, relative checkout locators and the pinned ranking CSVs. The source-audit tool can inspect archived evidence under `--archival-only`, which continues to record that the trigger needs separate reproduction. Its Java span extraction uses UTF-8 byte offsets after a tree-sitter `Point` crash on the local arm64 environment. This change is confined to the offline verifier.

### Closure coordinate mismatch

The pinned SBIR rank-9 overload `NodeUtil.nodeTypeMayHaveSideEffects(Node,AbstractCompiler)` occupies lines **1003–1024** in the exact upstream buggy revision. Its CSV lists **1005–1026**. Applying the pinned Defects4J production patch to the pinned fixed revision yields a separate benchmark reconstruction whose `NodeUtil.java` includes an extra blank line and comment. The CSV range matches that tree, and all 27 entries in the uncapped top-10 union mapped to a unique AST method there. The strict preflight used the exact upstream tree and rejected the mismatch. The reconstruction remains separate for review.

Closure's verified faulty overload is absent even from the uncapped 27-entry union. Its original CSV ranks were SBIR 35, Ochiai 34, and BoostN 126. Resolving the source provenance would therefore still leave a candidate-recall miss under the frozen schedule. The coordinate diagnostic preserves all original rankings.

## Mac preflight and token accounting

The Lang/Math run exported real CPU BGE-M3 at revision `694b61558aea4ae2512ed8d0e189d5cf8adc2259`. All eight checkpoint hashes matched the Phase 2.5 receipt. The saved bundles contain nontruncated 1,024-dimensional dense vectors and sparse weights for 92 Lang and 210 Math texts. Both arms used the same graph, schedule, embedding bundle, packing and allowances. Propagation converged for every target. Both arms stopped with the same mandatory nodes because complete mandatory prompts exceeded the proxy allowance. This provides no measurement of optional evidence selection at the preregistered exact-token allowance.

| Defect | Targets per arm | Complete mandatory prompt range in byte-proxy units | Configured allowance in the diagnostic | Per-arm conservative hold if all requests fit |
|---|---:|---:|---:|---:|
| Lang-54 | 15 | 2,738–2,891 | 1,024 | 125,280 tokens |
| Math-80 | 21 | 2,972–3,204 | 1,024 | 175,392 tokens |

All 36 target pairs failed packing under the current byte proxy. The full three-defect dry-run also recorded the Closure source error and exited 3. The byte proxy is system and user UTF-8 content bytes plus 256 units for template overhead. The BGE tokenizer recorded maxima of 623 and 810 tokens for encoder texts. Serving-template token IDs and provider usage are unavailable. No completed prompt token cost can be inferred from the proxy or encoder counts.

The running Ollama process points to the desktop app's **0.17.7** binary. The separate Homebrew CLI is **0.12.2**. The on-disk binary hashes, Llama3.1:8b model manifest digest and installed Go template hash are in the serving receipts. The GGUF Jinja template includes system content absent from the installed Go template's no-tools path. The public chat API documents token counts, and the versioned source passes tokenization through its runner. The exact served IDs for these requests remain **BLOCKED** pending a trusted capture from the matching serving path. [Ollama chat API](https://docs.ollama.com/api/chat), [Ollama v0.17.7 server source](https://github.com/ollama/ollama/blob/v0.17.7/server/routes.go).

The conservative hold remains `8,192 + 160 = 8,352` tokens per call. The 36,000-token arm cap allows at most four such holds in an offline envelope. The table shows the independent reservation barrier after prompt packing. No caps were changed. Resolving exact prompt tokenization and validating an admission contract are separate gates.

## Portability and tests

**PASS** for real-data relocation on the Mac. A copied Lang/Math input tree produced the same cohort lock, schedule, portable preparation identity, graph and bundle hashes, and complete paired selection records. The raw preparation identity changed with the host path. This run used the same launcher and scientific config with resource-location flags. **BLOCKED / unverified** for Linux/A100 execution and serving-tokenizer parity on that host. Real execution remains gated by the launcher.

The focused suite passed **35 tests in 13.96 seconds**. The regression suite passed **267 tests in 67.10 seconds** with connections restricted to local test fixtures. The new focused check exercises the archive-only source audit and verifies that it cannot promote archived evidence to a reproduced failure. No additional model or tool downloads were needed for these checks.

## Reproduce the diagnostics

From the repository root, use the existing project Python 3.11 and local Java 11. Substitute the local Ant library directory on Linux if replaying Closure. Each output directory is fresh. The trigger script records the full commands and logs. It runs one selected Java test method at a time.

```bash
graft_resources=data/graft_real_preparation
graft_results=results/graft_fl_real_preparation
graft_config=configs/graft_portable/real_mac_preflight.json
graft_pin=694b61558aea4ae2512ed8d0e189d5cf8adc2259
graft_bge="$HOME/.cache/graft-fl/models/BAAI/bge-m3/$graft_pin"
graft_ollama="$HOME/.ollama/models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home \
  python -B "$graft_resources/offline/reproduce_local_triggers.py" \
  --ant-lib /opt/homebrew/Cellar/gradle/9.0.0/libexec/lib \
  --output "$graft_resources/offline/trigger-replay-repeat"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -B -m rave_fl.graft.portable_experiment prepare-embeddings --profile local_mac \
  --config "$graft_config" --dataset-file "$graft_resources/online/validated-two-defects.json" \
  --bugs Lang-54,Math-80 --model-dir "$graft_bge" --revision "$graft_pin" \
  --embeddings-dir "$graft_resources/embeddings-repeat"

python -B -m rave_fl.graft.portable_experiment dry-run --profile local_mac \
  --config "$graft_config" --dataset-file "$graft_resources/online/diagnostic-three-defects.json" \
  --encoder-model-dir "$graft_bge" --embeddings-dir "$graft_resources/embeddings-repeat" \
  --generation-manifest "$graft_ollama/manifests/registry.ollama.ai/library/llama3.1/8b" \
  --ollama-models-dir "$graft_ollama" --output "$graft_results/mac-repeat-dry"
```

The expected dry-run exit is 3, with the blockers listed above. Use the saved `embeddings` directory to rerun planning without encoding. The `diagnostic-three-defects.json` manifest retains Closure's exact upstream input so the source rejection remains visible. The `validated-two-defects.json` manifest supports an explicitly restricted `--bugs Lang-54,Math-80` diagnostic. There is no authorized live run at this gate.

The reproducibility receipts are `data/graft_real_preparation/offline/{source-download-receipt,flexfl-download-receipt,trigger-validation,candidate-coordinate-diagnostic,closure-canonical-coordinate-check}.json`. Serving receipts are under `data/graft_real_preparation/offline/serving/`. The two- and three-defect plans and per-target evidence diagnostics are under `results/graft_fl_real_preparation/mac-real-{two,three}-defect-dry/`. `results/graft_fl_real_preparation/real-mac-relocation-audit.json` records the identity comparison. The pre-download report retains the focused test command and the Phase 6 report documents the exact-ID comparison interface.

## Remaining decisions

First obtain a trusted offline token-ID capture tied to Ollama 0.17.7, the installed template hash and the pinned GGUF. Any external source or toolchain acquisition requires its own scoped approval. Then validate budget admission for a complete schedule using serving-equivalent counts or a separately approved change to execution limits. Review Closure's exact upstream tree and the pinned benchmark reconstruction before choosing a source for real evaluation. Preserve the candidate-recall miss in either case. Run the same preflight on Linux/A100 before launching a benchmark there.
