# Phase 5 · Portable launcher and deterministic shards

12 November 2026. `GRAFT-FL`, baseline `e64d6dee9e642adaf08e4485512e7060f3335403`. **PASS for offline portability and synthetic execution.** Zero real LLM calls, additional model downloads or remote jobs. [Mac and Linux runbook](PORTABLE_RUNBOOK.md)

## Implementation

- The [portable launcher](../../../rave_fl/graft/portable_experiment.py) is one Python module for both operating systems. Its commands are `list`, `preflight`, `dry-run`, `prepare-embeddings`, `run`, `resume`, `report`, `plot` and `aggregate`. It uses the existing dataset adapters, fixed selector and runner, encoder, journal, result audit and figure renderer. The existing config and launcher overrides cover local resources, model, dataset, selected IDs, sample count and seed, budgets and output paths.
- The versioned cohort condition binds the frozen T0/T1 settings, implementation, prompt, dataset bytes and split, evaluation profile, generation settings, resource caps and tokenizer files. It binds every defect’s candidate pool, prior schedule, evidence graph, embedding bundle, paired payloads and reservations. The preregistration remains byte-identical. The [Mac profile](../../../configs/graft_portable/local_mac.json) and [Linux/A100 profile](../../../configs/graft_portable/remote_linux_a100.json) select a device placement and validate the OS. They cannot override scientific settings. Actual host placement is retained in the individual preflight and run manifests.
- Selection uses the adapter’s sorted or seeded order before limiting. Zero-based shards take every Nth selected ID and execute those members in canonical ID order. The lock verifies duplicate-free full coverage and the sum of the individual shard reservation ceilings. An explicit combined token and call cap is required when using multiple shards. The runner checks the prepared plan again before opening a run directory. It binds the cohort and shard index into its immutable manifest and checks label and split hashes. Journal reuse, unknown-cost refusal and exact accounting continue to use the existing execution code.
- Aggregate uses the Phase 4 read-only audit to validate each sealed run and reconstruct its ranking and accounting. It rejects missing or duplicate shards, foreign cohort membership, changed model/data/label/evaluator/tokenizer identities, changed candidate or prompt fingerprints, different provider identities, and duplicate paid request keys. It reports incomplete finalized pairs with their original reasons and denominators. The aggregate orders results by the frozen cohort regardless of CLI argument order.
- Preflight records host diagnostics, free disk, local tools and package availability, per-defect source/evidence identity, candidate and embedding preparation failures, and separate offline label and recall status. It supports explicit model-manifest and blob-size checks, optional metadata-only provider checks and local BGE-M3 layout checks. `prepare-embeddings` dispatches the existing offline CPU encoder, optionally over a single deterministic shard. Phase 5 did not load an encoder or a generation model.

## Verification

| Check | Result |
|---|---|
| Focused tests | **PASS**, 46 tests in 25.75 s, including nine new launcher cases. They exercise profiles, explicit and seeded selection, immutable conditions, cap and resource failures, encoder dispatch, overlap and missing shard rejection, paid resume, and unknown-cost refusal. |
| Guarded regression suite | **PASS**, 255 tests in 60.19 s, including the focused tests. The network guard allowed only the local HTTP fixture. |
| Two-profile synthetic rehearsal | **PASS**, byte-identical cohort locks, two disjoint shards, eight simulated requests and 3,343 simulated input-plus-output tokens. Both resumes made zero requests. Reversing aggregate input order reproduced all report files byte-for-byte. |
| 100-entry synthetic clone dry-run | **PASS**, 100 IDs, ten disjoint shards of ten, 400 planned calls, 816,100 conservative cohort token units under the explicit 1,000,000 cap, zero requests executed. The cloned inputs test software capacity and are synthetic repetitions. |
| Installed-model preflight | **PASS for local artifact checks**. The pinned `llama3.1:8b` manifest digest, blob presence and sizes, and BGE-M3 file layout passed without provider contact or weights loading. Exact serving-template tokens remain unverified. |
| Linux audit | **PASS** in a network-disabled Linux aarch64 container with Python 3.12.16. It validated both shard bindings, reconciled eight calls, reproduced aggregates, and checked the 100-ID lock. Both profile executions ran as synthetic rehearsals on the development Mac. |

The frozen runner’s only Phase 5 changes are cohort and plan checks around execution. Report generation reuses the same cohort definitions and plotting code for a shard or aggregate. Older run artifacts remain intact. Existing executable fingerprint checks still prevent appending to runs made by earlier implementations.

## Remaining gates

Real execution remains disabled, including when `--allow-inference` is supplied. The independent Defects4J inputs and labels, serving-tokenizer parity and actual Linux/A100 runtime remain open. The Linux container audit verifies portability of saved artifacts and conditions. A real GPU environment and provider run require later validation.

Keep one run directory per cohort shard and use that directory for resume. A fresh run is a separate execution. Independent hosts with separate copies of a cohort lock require manual assignment of shard indices. Duplicate sealed shards cannot be aggregated. Any host-path difference that changes the graph or rendered prompt requires resolving the input mismatch before merging.

## Retained artifacts and tests

`results/graft_fl_phase5_fixture/` holds the two profile preflights, frozen locks, sealed shards, merged reports and Linux audit. The large synthetic manifest and offline plans are under `large-input/` and `large-dry/`. `results/graft_fl_phase5_local_model_dry/preflight.json` records the local installed-model checks. The [runbook](PORTABLE_RUNBOOK.md) gives runnable examples and the [phase log](PHASE_LOG.md) records exact artifact digests.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B -m pytest -q tests/test_graft_phase5_portable.py tests/test_graft_phase4_reporting.py \
  tests/test_graft_phase3_admission.py tests/test_graft_fixed_evidence_*.py
```
