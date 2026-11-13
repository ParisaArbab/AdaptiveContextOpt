# Real-data preparation and approval gates

This document records the pre-download checkpoint. See the [post-approval readiness results](REAL_DATA_READINESS_RESULTS.md) for the acquired inputs, reproduced triggers, and Mac dry-run outcomes.

12 November 2026. Branch `GRAFT-FL`. **Offline checks passed. The three-defect Mac dry-run is blocked.** The scientific implementation and preregistration remain unchanged. No external resources were downloaded, no server or model was started, and no inference or benchmark executed.

## Missing resources and download scope

The requested first acquisition batch is the Defects4J framework and project metadata from `rjust/defects4j` at `8c16da8230843cdc918eaf4ddb449637f02b83c6`, the exact buggy and fixed Git objects for the three projects, and the nine SBIR/Ochiai/BoostN CSVs from `ParisaArbab/FlexFL_OriginalReplication` at `744143aa3e9c222b111beca55f5e75eab6d16285`. Each CSV already has a known Git blob ID in the local FlexFL tree. All nine blob bodies are missing. The [Phase 6 report](phase06.md) records Lang and Math revision SHAs. Closure’s exact revisions must come from the missing pinned metadata. The local `results/graft_fl_real_preparation/download-inventory.json` enumerates the exact CSV URLs, object IDs, known source revisions, and unavailable Closure blobs. Stage acquisitions separately from the existing reference clones and run archives.

Java 11.0.27 is installed locally. The sparse framework checkout lacks the runnable Defects4J CLI. `cpanm`, the standalone Ant command, and the declared Perl modules `DBD::CSV`, `JSON::Parse`, `String::Interpolate` and `Perl::Critic` were unavailable in the checked environment. The pinned framework must be inspected for its bundled and project-specific dependencies before requesting any additional runtime downloads. A blanket `init.sh` can download resources beyond the three target projects. The pinned BGE-M3 checkpoint and generation-model files are already installed.

| Gate | Status |
|---|---|
| Lang-54 | **BLOCKED.** Pinned revision mapping, archived failure and production patch available. Exact source objects, reproduced trigger, three ranking CSVs and independent AST gold verification remain open. |
| Math-80 | **BLOCKED.** Same resource gaps. The archived patch corroborates the target method signature. |
| Closure-61 | **BLOCKED.** Pinned metadata blobs, exact source revisions, trigger replay, ranking files and gold verification are missing. |
| Mac model and encoder artifacts | **PASS for local artifact checks.** The pinned Llama3.1 manifest, local blob sizes, BGE-M3 layout and saved real BGE-M3 bundle passed a one-defect synthetic-source dry-run. |
| Actual serving token IDs | **BLOCKED.** The re-audit reproduced the installed Ollama template and GGUF metadata. No trusted exact serving token-ID capture is available. The GGUF Jinja template remains a separate rendering. |
| Real-data dry-run | **BLOCKED.** The launcher wrote an explicit missing-manifest diagnostic. No cohort lock or online data was fabricated. |
| Real execution and Linux/A100 | **BLOCKED / unverified.** The launcher still gates real inference. |

## Independent source and gold checks

The [offline verifier](../../../scripts/verify_graft_real_sources.py) checks the declared source SHA against pinned Defects4J Git metadata and reuses the fixed checkout and FlexFL candidate pipeline. It verifies CSV bytes against the pinned FlexFL Git objects, checks each span and overload against the buggy Java AST, and derives affected buggy methods from the zero-context buggy-to-fixed production diff. Independent gold must agree with the pinned historical labels. Missing objects, ambiguous methods, a diff that touches production lines outside a unique method, or a candidate mismatch block the source audit. The candidate report and derived labels are separate. The online ranking tree excludes the FlexFL ground-truth directory. This verifier records that trigger reproduction is a separate gate.

After approved provisioning, use distinct exact-source and test-harness worktrees. Confirm the original production source matches between them before reusing the observed log. Retain the test command, selected trigger, exit status, stdout/stderr and the newly produced `failing_tests` file. Check the failing test identifier against pinned metadata, and verify the source remained clean. For example, once Defects4J and its pinned project dependencies are available, the bounded Lang trigger commands are:

```bash
graft_trigger=data/graft_real_preparation/trigger_worktrees/Lang-54
mkdir -p data/graft_real_preparation/trigger_worktrees data/graft_real_preparation/offline/logs
defects4j checkout -p Lang -v 54b -w "$graft_trigger"
defects4j test -w "$graft_trigger" \
  -t org.apache.commons.lang.LocaleUtilsTest::testLang328 \
  > data/graft_real_preparation/offline/logs/Lang-54-test.log 2>&1
graft_test_exit=$?
printf '%s\n' "$graft_test_exit" > data/graft_real_preparation/offline/logs/Lang-54-test.exit
```

Use `Math`, `80b`, and `org.apache.commons.math.linear.EigenDecompositionImplTest::testMathpbx02` for Math. Resolve Closure’s test from the pinned trigger blob before running it. Keep patch and fixed-source evidence outside the online manifest. In a separate label-free source-audit config, set `labels_file` to null and use the reviewed manifest of exact buggy checkouts, explicit production source roots, observed failure logs and CSV-only FlexFL data. Then run:

```bash
graft_resources=data/graft_real_preparation
graft_d4j_pin=8c16da8230843cdc918eaf4ddb449637f02b83c6
graft_flexfl_pin=744143aa3e9c222b111beca55f5e75eab6d16285
for graft_bug in Lang-54 Math-80 Closure-61; do
  python -B -m scripts.verify_graft_real_sources --config "$graft_resources/source-audit.json" \
    --bug "$graft_bug" --metadata "$graft_resources/references/defects4j" \
    --metadata-revision "$graft_d4j_pin" --flexfl-reference "$graft_resources/references/flexfl" \
    --flexfl-revision "$graft_flexfl_pin" --historical-labels "$graft_resources/references/flexfl" \
    --output "$graft_resources/offline/source-audit/$graft_bug" || break
done
```

After inspecting the source and trigger receipts, use the three independently verified label rows in `offline/labels.json`. Keep the online manifests and resources relative to their roots for relocation. The [Mac preflight configuration](../../../configs/graft_portable/real_mac_preflight.json) uses the preregistered candidate schedule, model identity, output cap, fusion and planning limits. It references `data/graft_real_preparation/online/instances.json`. After preparing the manifest and source audits, export with the pinned CPU encoder and invoke the same portable CLI:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
graft_config=configs/graft_portable/real_mac_preflight.json
graft_bge_revision=694b61558aea4ae2512ed8d0e189d5cf8adc2259
graft_bge="$HOME/.cache/graft-fl/models/BAAI/bge-m3/$graft_bge_revision"
graft_ollama="$HOME/.ollama/models"
python -B -m rave_fl.graft.portable_experiment prepare-embeddings --profile local_mac \
  --config "$graft_config" --model-dir "$graft_bge" --revision "$graft_bge_revision"
python -B -m rave_fl.graft.portable_experiment dry-run --profile local_mac \
  --config "$graft_config" --encoder-model-dir "$graft_bge" \
  --generation-manifest "$graft_ollama/manifests/registry.ollama.ai/library/llama3.1/8b" \
  --ollama-models-dir "$graft_ollama" --output results/my_real_mac_dry
```

## Token accounting and admission blockers

Use the [Phase 6 exact reference comparison](phase06.md) with a trusted capture from the pinned serving build to compare full token ID sequences. Byte proxies, local-template token IDs and provider-reported usage remain separate fields. The 1,024 value in the current config uses byte-proxy units while `tokenizer_dir` is null. A synthetic-source diagnostic found mandatory complete inputs of 1,297 and 1,288 proxy units at that allowance. This demonstrates a potential packing blocker and does not measure the target defects. The 2,048-unit synthetic-source diagnostic passed as a separate fixture condition.

The default live-backend reservation remains 8,352 tokens per request at context 8,192 and output cap 160. The 36,000-token per-arm cap covers at most four such full holds. A 24-target arm would reserve 200,448 tokens under that bound. No schedule was shortened and no cap was raised. A validated live reservation contract or an explicitly approved change to execution caps is still needed for larger candidate pools. Exact token parity alone does not change the current reservation logic.

The local artifacts are under `results/graft_fl_real_preparation/`. The real-data dry-run returned exit 3 with the missing-manifest diagnostic. No live run command is enabled at this gate.

The focused suite passed **34 tests in 12.75 s**. The fixture-only guarded regression suite passed **266 tests in 63.24 s**. New synthetic checks cover pinned ranker bytes, separate gold output, rejection of a gold directory in the online ranker tree, wrong-overload spans, and patch hunks outside a unique method. From the project Python environment, reproduce the focused suite with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m pytest -q \
  tests/test_graft_real_preparation.py tests/test_graft_phase6_readiness.py \
  tests/test_graft_phase5_portable.py tests/test_graft_phase3_admission.py
```
