# Portable GRAFT-FL runbook

Run these commands from the repository root using the project’s Python environment. The same entry point runs on macOS and Linux. Install the project’s evaluation dependencies in that environment before using the launcher. The encoder command also needs the Phase 2.5 embedding dependencies and an existing local checkpoint.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
graft() { python -B -m rave_fl.graft.portable_experiment "$@"; }
graft list
graft preflight --help
```

Use `local_mac` on macOS and `remote_linux_a100` on Linux. The profile controls device placement and may set an endpoint or timeout in a local profile JSON. The profile name records intent. Preflight records the actual OS and whether `nvidia-smi` is available. `--simulate-host` permits an OS mismatch exclusively with the fake backend.

## Small Mac rehearsal

This uses the retained synthetic fixture with saved BGE-M3 bundles. Use a fresh output directory. The separate 24,000-token and eight-call limits apply across both shards.

```bash
graft_config=results/graft_fl_phase4_final/config.json
graft_out=results/my_graft_small
graft preflight --profile local_mac --config "$graft_config" \
  --bugs covered-java,covered-python --shard-count 2 \
  --cohort-physical-budget 24000 --cohort-max-new-calls 8 --output "$graft_out/pre"
graft run --profile local_mac --config "$graft_config" \
  --cohort "$graft_out/pre/cohort.json" --shard-index 0 --output "$graft_out/shard-0"
graft run --profile local_mac --config "$graft_config" \
  --cohort "$graft_out/pre/cohort.json" --shard-index 1 --output "$graft_out/shard-1"
graft aggregate --cohort "$graft_out/pre/cohort.json" \
  --run "$graft_out/shard-0" --run "$graft_out/shard-1" --output "$graft_out/merged"
```

The report includes `REPORT.md`, `summary.json`, three CSVs and PNG/SVG figures. `report` and `plot` both regenerate a complete report from a sealed shard without a dataset or model.

```bash
graft report --run "$graft_out/shard-0" --output "$graft_out/shard-0-report"
graft resume --profile local_mac --config "$graft_config" \
  --cohort "$graft_out/pre/cohort.json" --shard-index 0 --output "$graft_out/shard-0"
```

Use the same config, host settings and output directory for resume. Keep one execution directory per cohort shard. A fresh directory is a separate execution with its own caps. The journal blocks resume after unknown-cost or pending requests and keeps their holds. Investigate the preserved journal, raw ledger and accounting before taking further action. Two independently launched copies of a shard are rejected at aggregation. The launcher does not coordinate execution across hosts with separate copies of a cohort lock.

## Linux and larger selections

Copy the checkout and fixture or verified dataset resources to Linux. Set local dataset, checkout, label and embedding paths in the config. Relative config paths are evaluated from the repository root. Relative checkout and evidence paths inside a normal `instances` manifest are evaluated from that manifest’s directory. Keep the same manifest bytes and artifact contents when sharing a cohort across hosts. The fixed runner identifies an absolute failure-log locator by its content hash in the graph and retains the local locator in the exact resume manifest. It rejects a known local checkout or log path embedded in the supplied failure text. Use stable source identifiers and relative resource paths when preparing a new portable manifest. For bundles built against earlier graphs with absolute locators, export against the updated graph and create a fresh cohort lock. Saved archives remain available for reporting.

The following dry-run exercises the retained 100-entry synthetic clone manifest. It executes zero requests. To use a different prepared manifest, change `--config`, `--dataset` and `--dataset-file` as appropriate. Choose explicit shard and cohort limits for that input.

```bash
graft_config=results/graft_fl_phase5_fixture/large-input/config.json
graft_out=results/my_graft_linux_dry
graft dry-run --profile remote_linux_a100 --config "$graft_config" --all \
  --sample random --seed 17 --shard-count 10 --max-new-calls 40 --physical-budget 100000 \
  --cohort-physical-budget 1000000 --cohort-max-new-calls 400 --output "$graft_out"
```

For a small Linux synthetic rehearsal, use the Mac example with the fixture config relocated to Linux and replace `--profile local_mac` with `--profile remote_linux_a100`. For a larger future run, change `--limit` to the desired sample size or use `--all`. Review every `preflight.json` inventory status and inspect the `plans/` files before running any shard. Use `--bugs A,B` or `--bug-file ids.txt` for an explicit cohort. Use the same seed, budget and model overrides for execution. The shard index is zero-based. `aggregate` requires exactly one sealed run for every shard in the cohort, irrespective of the order of `--run` arguments.

## Switch resources or model

Keep a separate JSON config for each experimental condition. Pass generation options through flags or the `benchmark` object. Supported dataset adapter names are listed by `graft list`. The fixed graph pipeline requires Java or Python method/function candidates, a clean buggy checkout, and observed failure evidence. `--check-provider` explicitly reads Ollama model metadata. Offline checks can instead read the installed manifest and verify blob presence and sizes using `--generation-manifest` and `--ollama-models-dir`.

```bash
graft preflight --profile local_mac --config results/graft_fl_phase4_final/config.json \
  --bugs covered-java,covered-python --dataset custom \
  --dataset-file results/graft_fl_phase4_final/instances.json \
  --backend ollama --endpoint http://127.0.0.1:11434 --model llama3.1:8b \
  --model-digest 46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e \
  --embeddings-dir results/graft_fl_phase4_final/embeddings \
  --output results/my_graft_model_preflight
```

Supply the new dataset’s own offline `--labels-file`, `--embeddings-dir` and optional `--tokenizer-dir`. Point to existing BGE-M3 weights with `--encoder-model-dir` to check their layout. Prepare missing embeddings in a separate encoder-only step before generating the cohort lock. The encoder uses CPU FP32 and rejects an existing bundle path.

```bash
graft_revision=694b61558aea4ae2512ed8d0e189d5cf8adc2259
graft prepare-embeddings --profile local_mac --config results/graft_fl_phase4_final/config.json \
  --bugs covered-java,covered-python \
  --embeddings-dir results/my_graft_embeddings \
  --model-dir "$HOME/.cache/graft-fl/models/BAAI/bge-m3/$graft_revision" --revision "$graft_revision"
```

`graft <command> --help` lists all model, endpoint, selection, tokenizer, embedding, token and call-cap overrides. Exit `0` means success, `3` preserves a blocked preflight or incomplete pair, and `2` indicates an input, integrity or execution error. Phase 5 supports real-backend planning only. `run` and `resume` still refuse real execution even with `--allow-inference`. Serving-tokenizer validation, verified independent Defects4J inputs and actual Linux/A100 execution remain later gates.
