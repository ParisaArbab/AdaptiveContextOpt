# Phase 6 · Provider-free real-evaluation readiness

12 November 2026. Branch `GRAFT-FL`, baseline `e64d6dee9e642adaf08e4485512e7060f3335403`. **Offline engineering checks passed. Real evaluation is blocked.** The scientific preregistration, selector, ranking and one-flag T0/T1 comparison are unchanged. Zero model calls, weight tensor loads, new model downloads, remote jobs or benchmark executions.

## Defects4J resource audit

The [offline audit utility](../../../scripts/audit_graft_phase6.py) read committed blobs from the local Defects4J reference at `8c16da8230843cdc918eaf4ddb449637f02b83c6`, with Git network protocols and lazy fetching disabled. The local clone records the official Defects4J remote. Each available blob was verified against its pinned Git object and saved with a SHA-256. The archived trigger traces are historical observations from the pinned reference. No target failure was rerun in this phase.

| Defect | Pinned metadata available | Independent local verification still required |
|---|---|---|
| Lang-54 | Buggy `77a3c20bc0f30ca44f30aadad7f8ef0bc2528d0b`, fixed `75d931a3264b73caa9cdd7d3373375cc33008ddf`. Archived `LocaleUtilsTest::testLang328` failure and source patch verified. The trace corroborates the historical FlexFL label `LocaleUtils.toLocale(String)`. | Clean exact buggy checkout, test replay, matching SBIR/Ochiai/BoostN pool, source-span resolution and offline gold verification. |
| Math-80 | Buggy `c78a0e3b003ac36d57a3c895fc8240ef66546a56`, fixed `83f18d5299fbadf91ec792c95ee96bcd916ca9af`. Archived `EigenDecompositionImplTest::testMathpbx02` failure and source patch verified. The patch shows the historical label’s `flipIfWarranted(int,int)` declaration and changed line. | Clean exact buggy checkout, test replay, matching ranker pool, source-span resolution and offline gold verification. |
| Closure-61 | The pinned Git tree names the revision table, trigger trace and source patch blobs, but all three blobs are missing locally. FlexFL lists `NodeUtil.functionCallHasSideEffects(Node,AbstractCompiler)` as a historical label. | Missing Defects4J metadata and all source, failure, candidate and gold verification. |

No target checkout or corresponding candidate CSVs were found in the checked local roots. The installed Java executable reports OpenJDK 17. Defects4J was unavailable on PATH, and the available Lang-1 and Math-1 repositories are shallow and do not contain the target commit objects. The verifier did not substitute earlier development defects. Patch files and historical labels were kept separate from online inputs. No ready Defects4J manifest was written.

For a subsequent approved provisioning step, obtain the pinned Closure metadata and all three exact buggy checkouts and confirm clean tracked sources at the recorded revision. Reproduce each trigger using the project-compatible Defects4J and Java environment, retaining the command, exit status, full failure log and checksums. Resolve every candidate span against the same checkout with the preregistered FlexFL provider. Independently map the official production patch to the buggy method’s full signature and keep the resulting labels in a separate evaluation file. A candidate miss remains a measured recall outcome once the labels and pool are verified. Missing candidates or source identity prevent declaring the input ready.

## Serving template and token counts

The installed `llama3.1:8b` manifest matched digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`. The installed Ollama template bytes matched SHA-256 `948af2743fc78a328dcb3b0f5a31b3d75f415840fdb699e8b1235978392ecf85`. The offline GGUF inspection read 7,823,282 metadata bytes and zero tensor bytes. It found the `llama-bpe` pre-tokenizer, 128,256 vocabulary entries, BOS ID 128000 and EOS ID 128009. The GGUF-embedded Jinja template includes date text in the system prompt. The installed Ollama Go template places its knowledge date stanza inside the tools branch, which is inactive for the paired no-tools request. The embedded template therefore cannot establish Ollama parity for this experiment.

**Serving-token parity remains blocked.** No trusted serving-build token-ID capture exists locally for the exact fixed requests. The comparison command checks exact IDs rather than counts alone. Its reference must pin the model manifest, serving-template SHA and serving build, identify an offline serving tokenizer or archival token-ID capture, and provide exact IDs for every distinct planned request keyed by the canonical request fingerprint. It reports complete-message byte-proxy units, local template counts, reference ID counts and optional provider-reported input usage separately. A supplied-reference match records that comparison only. The live context and reservation bounds remain conservative. Embedding-tokenizer counts stay separate.

## Portability and protocol checks

The [fixed input identity layer](../../../rave_fl/graft/fixed_inputs.py) replaces absolute evidence locators with a content-addressed graph source ID. The raw host locator remains part of the local prepared identity. The portable prepared identity removes host checkout, candidate-file and FlexFL directory locators while retaining content, prior scores, extracted spans and candidate provenance. The cohort continues to bind the graph, bundle and exact request. The encoder launcher and selector use the same graph identity. This fixes relocation of identical file-backed evidence without rewriting its text. A known local path embedded in the failure text blocks planning. Evidence byte changes invalidate the bundle. Manifests with absolute paths can still differ by bytes after moving and require a new cohort or an identically relocated manifest with relative references.

The Mac relocation test compared full prepared plans from two mounts with the same file-backed synthetic evidence. Raw preparation fingerprints differed and portable cohort conditions, requests and graph fingerprints matched. A network-disabled Linux aarch64 container reproduced the graph digest and both rendered Python fixture targets under different mount paths. Linux/A100 provider execution and real CUDA encoding remain unverified.

Separate offline preflights switched between the installed `llama3.1:8b` and `qwen3-vl:8b` manifests using configuration only. Each selected one prepared fixture defect and used identical frozen T0/T1 settings. Model identities remained separate cohort conditions. A synthetic Java integration test also switched the adapter, sample size and device through configuration. Phase 5 host-profile tests remained in the regression suite.

## Validation and reproduction

The focused suite passed **29 tests in 15.04 s**. The fixture-only guarded regression suite passed **261 tests in 67.21 s**, including the focused checks. The new tests covered graph and cohort relocation, changed-log refusal, embedded local paths, configuration-only switching, pinned Git blob extraction, bounded metadata reads, and equal-length token sequences with different IDs. The two archive audits reproduced byte-for-byte in fresh outputs.

Run from the repository root in the project environment. Both readiness commands write the audit and exit 3 while the real gate is blocked. Neither contacts a provider.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
graft_phase6_out=$(mktemp -d)
python -B -m scripts.audit_graft_phase6 defects4j \
  --metadata references/defects4j_pilot_meta \
  --revision 8c16da8230843cdc918eaf4ddb449637f02b83c6 \
  --flexfl references/FlexFL_Time25 --checkouts data/rave_fl_checkouts \
  --output "$graft_phase6_out/defects4j" || test "$?" -eq 3
graft_model_root="$HOME/.ollama/models"
python -B -m scripts.audit_graft_phase6 serving \
  --manifest "$graft_model_root/manifests/registry.ollama.ai/library/llama3.1/8b" \
  --models-dir "$graft_model_root" \
  --digest 46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e \
  --output "$graft_phase6_out/serving" || test "$?" -eq 3
python -B -m pytest -q tests/test_graft_phase6_readiness.py \
  tests/test_graft_phase5_portable.py tests/test_graft_phase3_admission.py
```

For a later accepted exact token reference, run `python -B -m scripts.audit_graft_phase6 compare --help` for the required artifacts. The retained evidence, installed-template audit, two separate model preflights and Linux audit are under `results/graft_fl_phase6/`. [Phase log and recorded digests](PHASE_LOG.md)
