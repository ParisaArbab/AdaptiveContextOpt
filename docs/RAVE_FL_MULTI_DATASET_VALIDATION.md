# RAVE-FL Multi-Dataset Infrastructure Validation

Validation date: 2026-10-26. Development branch: `rave-fl-adaptive-pilot`. Host: macOS 26.1 ARM64, Python 3.11.0. These are engineering checks, not new research accuracy results.

## What was executed

- **132 targeted tests passed:** 41 new universal evaluation cases and 91 existing RAVE-FL regression cases.
- Synthetic Java/Python integration: two tiny committed fixture sources, five default policies, four fusions, **40 completed arms**. Six unique fake requests produced 1,204 simulated provider tokens and **zero real inference tokens**.
- Resume of that run: six total journaled fake requests remained six; no duplicate request was paid or simulated.
- A second synthetic run with an 18K rather than 16K policy budget reused the first run's exact shared cache: 40 completed arms, **zero new provider calls**. Its comparison produced 80 completed matched file/method pairs and budget-sensitivity tables/figures. This validates plumbing, not policy superiority.
- Exact historical Time-25 replay through the new Bash CLI: Fixed L1 reproduced 11,298 verification tokens/rank 4; Highest Suspicion and V2 Utility reproduced 12,516/rank 1. **Zero new inference calls.** These are the original Llama observations, not fresh measurements.
- Frozen-observation reranking: four fusions applied to the recorded Time-25 Fixed L1 state with no provider contact or new action purchase. Oracle actions were not used.
- macOS setup/preflight ran directly, using fake-provider metadata for installation-interface validation. A100 setup passed portable checks; NVIDIA metadata was additionally mocked in the focused test. No A100 experiment, GPU allocation, model pull, benchmark download, external repository test suite or real LLM inference was executed.
- `bash -n`, Python compilation/import checks, all four evaluation YAML configuration validations and source whitespace checks passed. ShellCheck was not installed. The installed Black 19.3b0 CLI has a Click incompatibility; its formatting library API successfully formatted and AST-checked the new Python files without installing or changing dependencies.
- **357 tracked historical result/report files** matched the original Git HEAD. Their SHA-256 inventory is retained in `results/rave_fl_eval_validation/historical_preservation.json`. No old research report, raw ledger or manifest was regenerated.

The targeted suite excludes unrelated untracked `tests/test_pipeline_dependencies.py` and external benchmark test suites. It does not assert that every repository test or external toolchain passes. Existing unrelated `.tools/` and `data/repos/` work was preserved.

## Focused checks covered

Typed schema and native adapter parsing; exclusion of gold patches/PR hints/post-fix logs from online objects; Python class/function and Java AST extraction; candidate coverage failures without gold insertion; explicit observed-evidence requirements; imported span bounds; missing models; inference authorization and quoted-boolean rejection; all controller names/fusions; unavailable disagreement inputs and a successful imported-ranker disagreement case; malformed paid responses; global/per-bug admission; concurrent identical request reuse; unknown/pending usage without retry; paid provider-bound violations; cache source/settings/digest invalidation; immutable resume and checksum validation; file/method/multiple-location metrics; incomplete and explicitly skipped attempts; frozen mode without provider access; cross-run compatibility checks; split disjointness/empty-selection safety; project grouping and development-only Time-25; independent-defect paired bootstrap safeguards; CLI/report generation; mocked A100 preflight; historical saved-observation regression.

## Exact test command

```bash
python3 -m pytest -q \
  tests/test_rave_eval.py \
  tests/test_rave_fl.py tests/test_rave_fl_adapter.py \
  tests/test_rave_fl_evidence.py tests/test_rave_fl_expanded.py \
  tests/test_rave_fl_iteration_d.py tests/test_rave_fl_phase3.py \
  tests/test_rave_fl_pilot.py tests/test_rave_fl_rank_repair.py tests/test_rave_fl_v3.py
bash -n scripts/rave_fl_run.sh scripts/rave_fl_setup_mac.sh scripts/rave_fl_setup_a100.sh
python3 -m compileall -q rave_fl/eval
```

## Retained artifacts

All paths are relative to the repository root. The complete generated fixture/cache/SQLite/figure trees are local ignored artifacts; committed validation summaries and logs identify them. Do not recursively add the fixture's embedded `.git` checkout or large caches to this repository.

| Artifact | Path |
|---|---|
| Structured validation summary | `results/rave_fl_eval_validation/validation.json` |
| Test log | `results/rave_fl_eval_validation/tests.log` |
| Historical preservation inventory | `results/rave_fl_eval_validation/historical_preservation.json` |
| Synthetic fixture manifest and tiny repository | `results/rave_fl_eval_validation/fixture/instances.json` |
| Final synthetic run | `results/rave_fl_eval_validation/final/synthetic/` |
| Shared-cache second run | `results/rave_fl_eval_validation/final/cached/` |
| Saved-run comparison | `results/rave_fl_eval_validation/final/comparison/` |
| Exact Time-25 CLI replay | `results/rave_fl_eval_validation/final/time25/` |
| Frozen Time-25 fusion ablation | `results/rave_fl_eval_validation/final/frozen/` |
| Adapter/policy capabilities | `docs/rave_fl_dataset_capabilities.json` |

Each run has a manifest/config/environment, source hashes, status records, candidate inputs, per-arm results, offline-only labels, raw ledger/journal, accounting audits, CSV tables, PNG/SVG figures, artifact index and an English report. The final synthetic run emits 27 PNG/SVG figure pairs; exact Time-25 replay emits 22, and the frozen ablation emits 21. Missing GPU measurements and inappropriate statistical intervals remain unavailable.

## Reproduce the fixture evaluation

Use a new output directory each time. For an entirely fresh fixture, first run `python3 -m rave_fl.eval.fixtures --output-dir /tmp/rave-fixture-new` and point `--dataset-file` to its `instances.json`.

```bash
bash scripts/rave_fl_run.sh run --dataset custom \
  --dataset-file results/rave_fl_eval_validation/fixture/instances.json \
  --backend fake --model fake:fixture --context-window 4096 --admission strict \
  --candidate-count 10 --budget 16000 --max-full 1 \
  --policies full,fixed_l1,highest_suspicion,v2_utility,uncertainty_aware \
  --fusions shared_prior_evidence_fusion_v1,consistent_score,rrf,prior_aware \
  --physical-budget 500000 --per-bug-physical-budget 100000 \
  --output-dir /tmp/rave-fixture-run-new --run-id fixture-reproduction
bash scripts/rave_fl_run.sh resume --run-dir /tmp/rave-fixture-run-new
bash scripts/rave_fl_run.sh report --run-dir /tmp/rave-fixture-run-new
bash scripts/rave_fl_run.sh run \
  --config /tmp/rave-fixture-run-new/reproduction_config.json \
  --budget 18000 --cache-dir /tmp/rave-fixture-run-new/cache \
  --output-dir /tmp/rave-fixture-cached-new --run-id cached-reproduction
bash scripts/rave_fl_run.sh compare \
  --runs /tmp/rave-fixture-run-new,/tmp/rave-fixture-cached-new \
  --output-dir /tmp/rave-fixture-comparison-new
```

The central [evaluation guide](RAVE_FL_MULTI_DATASET_EVALUATION.md) contains exact historical replay, frozen fusion, Mac/A100 preflight, and future live commands for three Defects4J bugs, 20 Lite issues, 50 Verified issues and 20 native Multi-SWE Java instances.

## Implementation and integration status

Added `rave_fl/eval/` contracts, adapters, candidate providers, language representations, provider/journal/cache, orchestration, policy wrappers, fusion strategies, metrics, preflight, reporting/compare, fixtures and project split utilities. Added three executable Bash scripts, four YAML configurations, dependency list, capability matrix, guides and focused tests. Modified only optional fusion/order/cost-gate hooks in three historical controller modules; their default behavior and historical CLI paths remain intact. README links the new guide.

No newly trained model, controller optimization, independent real-defect evaluation or generalization claim was introduced. Alternative fusions are transparent uncalibrated ablations. Native Multi-SWE Java parsing is implemented and fixture-tested; that does not establish that an official external Multi-SWE harness ran.

Remaining external requirements: selected local exports/checkouts and immutable revisions, official buggy identity/evidence preparation, reliable gold labels, Java parser dependencies, model/provider setup and optional localizer/semantic plugins. Imported span bounds do not prove arbitrary identifiers are real methods. Reference patch file labels remain filtered silver proxies. Other languages support registered file/Full paths only. Actual A100 placement, peak memory and model throughput require later hardware validation. Preflight reports physical host RAM and disk plus optional device memory; it does not guarantee that a model will fit or that the physical RAM total is currently allocatable free RAM.

No automatic retries are made for uncertain paid calls. Physical spending limits are scoped to one run directory; separate concurrent jobs require external coordination. Tight-cap multi-worker studies may have scheduling-dependent allocation. The default single worker is recommended for matched deterministic comparisons. These limits are documented rather than hidden by dummy observations.

The remote lookup during validation confirmed `origin/rave-fl-adaptive-pilot` at `95ffdc4e2b677c2b856d75f012c4267277bc94cf` and the parallel collaborator branch at `e6e9fd9ffde1c396c347f2c7322ca3606534749b`. No remote refs were changed. Development changes are committed locally on the requested branch; no collaborator lineage was merged or overwritten. The precise future integration plan is in the central guide.
