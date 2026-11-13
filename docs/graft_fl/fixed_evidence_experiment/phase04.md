# Phase 4 · Synthetic rehearsal and reproducible reporting

12 November 2026. `GRAFT-FL`, baseline `e64d6dee9e642adaf08e4485512e7060f3335403`. **PASS for synthetic execution and offline reporting.** The preregistration, fixed runner, selector, prompts, allowance, schedules, model settings, output caps and fusion remain frozen. The two arms differ only by propagation and its effect on evidence. No model weights were loaded or downloaded and no real LLM inference was executed.

## Implementation

- The [offline reporter](../../../rave_fl/graft/fixed_evidence_reporting.py) reads finalized results, the manifest, accounting snapshot, journal and raw request ledger. It reconciles the exact scheduled targets, request payloads, verdicts and counters, checks paired prompt equivalence and reconstructs the frozen RRF rankings from the saved observations. It uses the sealed method metrics from the selected evaluation profile. Dataset, labels, embedding files and model endpoints are unnecessary for report regeneration. Output goes to a separate directory, and the reader opens only a closed, checkpointed journal. Unresolved or unfinished runs retain their raw accounting and cannot enter a finalized report.
- The [figure renderer](../../../rave_fl/graft/fixed_evidence_figures.py) produces headless 300 dpi PNG and vector SVG charts for accuracy, token costs, accuracy versus cost and Pareto status, per-defect differences, and evidence selection. Each figure labels the synthetic condition and identifies the relevant counter or token definition. Empty cohorts receive an explicit unavailable panel.
- The [rehearsal driver](../../../scripts/rehearse_graft_fixed_evidence.py) reused the prepared Java/Python fixture and the saved real BGE-M3 bundles. It prepared six named fixture cases with exactly two scheduled targets each, using the existing fake provider, journal and ranking pipeline. It injected one invalid verdict at a known fixture request and ran a separate one-call-per-arm limit case. Gold remained in a separate offline manifest. Shared external cache stayed off and exact journal reuse stayed available under the frozen protocol.

## Rehearsal results

**SYNTHETIC SOFTWARE TEST — NOT RESEARCH RESULTS**

| Operational measure | T0 | T1 |
|---|---:|---:|
| Complete arms | 5 of 6 | 6 of 6 |
| Settled synthetic requests | 12 | 12 |
| Invalid paid replies | 1 | 0 |
| Simulated provider input | 4,362 | 5,038 |
| Simulated provider output | 324 | 324 |
| Logical and settled simulated physical tokens | 4,686 | 5,362 |
| Real physical tokens | 0 | 0 |

There were five jointly complete pairs. The invalid pair's usage was retained and its paired accuracy was unavailable. The 24 settled calls consumed 10,048 simulated input-plus-output tokens. Resume made zero new requests. The separate infeasible pair made zero requests and had no accuracy denominator. The explicit fixture run cap was 60,000 simulated tokens. The Phase 3 planner computed a serial upper bound of 49,672 and a sum of full holds of 102,144. Those values are planning bounds.

The primary matched cohort had three joint-complete pairs with evaluable method labels. Each arm had Top-1, Top-5 and MRR of 0.667. Two pairs had full candidate recall. The third had known gold outside the candidate pool and contributed zero end-to-end accuracy. The full-recall sensitivity cohort therefore had two pairs, each arm with MRR 1.0. A missing-label case and a file-only-label case had unavailable method accuracy. The invalid case kept its signed cost difference. The two accuracy denominators and simulated costs exercise report behavior on fixture data.

## Validation and gates

| Check | Result |
|---|---|
| Focused suite | **PASS**, 44 tests in 21.66 s. Nine new cases cover sealed offline regeneration, empty cohorts, candidate mismatch, ranking/usage/ledger tampering, synthetic-only reporting, unresolved usage and strict Java overload matching. Existing tests cover label isolation, interrupted resume and strict cap boundaries. |
| Guarded regression suite | **PASS**, 246 tests in 56.90 s, including the focused tests. The socket guard allowed only the in-process HTTP fixture. |
| Artifact regeneration | **PASS**, independent invocations produced byte-identical Markdown, JSON, CSV, PNG and SVG files in the tested environment. Tests also removed access to the prepared inputs and labels before regeneration and verified the run files were byte-identical afterward. The reporter regenerated a Phase 3 archive without resuming it. |
| Linux audit | **PASS** for offline accounting and aggregation. A network-disabled Linux aarch64 container with Python 3.12.16 matched six pairs, 24 settled calls, 12 evidence targets and the Mac aggregates. Rendering was validated on macOS with Matplotlib 3.10.6 and NumPy 2.3.3. |
| Real evaluation | **OPEN**. The independent Defects4J checkouts, observed failure evidence, validated candidates and labels, serving-tokenizer parity, and Linux/A100 execution remain separate gates. The reporting CLI currently accepts sealed synthetic results. |

The proposed Phase 5 scope is the offline portable launcher, host profiles, configurable manifests and deterministic sharding and aggregation. It requires separate approval.

## Reproduction

Run from the repository root with the project interpreter. The fixture driver requires an explicit run cap and uses a new output directory. Its `config.json` and `denied-config.json` record the two sets of limits. The fixed runner continues to accept dataset, model, bug selection, paths and resource limits through configuration.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
graft_phase4_out=results/graft_fl_phase4_reproduction
python -B -m scripts.rehearse_graft_fixed_evidence \
  --fixture-config results/graft_fl_phase25_fixture/config.json \
  --fixture-labels results/graft_fl_phase3_fixture/fixture-labels.json \
  --physical-budget 60000 --output "$graft_phase4_out"
PYTHONHASHSEED=19 python -B -m rave_fl.graft.fixed_evidence_reporting \
  --run "$graft_phase4_out/run" --output "$graft_phase4_out/report-reproduced"
diff -r "$graft_phase4_out/report" "$graft_phase4_out/report-reproduced"
python -B -m pytest -q tests/test_graft_phase4_reporting.py tests/test_graft_phase3_admission.py \
  tests/test_graft_fixed_evidence_*.py tests/test_graft_evaluation.py
```

The retained rehearsal is under `results/graft_fl_phase4_final/`. `run/` preserves the sealed execution, `report/` contains `REPORT.md`, `summary.json`, `per_bug_arms.csv`, `paired_differences.csv`, `evidence_selection.csv` and all five figures in both formats. `denied-run/` and `denied-report/` retain the budget rejection. `linux-audit.json` and `rehearsal.json` record the Linux and resume checks. These local result folders are ignored by Git. The [phase log](PHASE_LOG.md) records checksums and the regression procedure.
