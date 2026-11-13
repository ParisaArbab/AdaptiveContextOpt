# Phase 3 · Matched prompts, admission, and accounting

12 November 2026. `GRAFT-FL`, baseline `e64d6dee9e642adaf08e4485512e7060f3335403`. **PASS for offline prompt validation and synthetic accounting.** Zero real LLM calls. No encoder or generation weights were loaded. The dry plans reused the real Phase 2.5 BGE-M3 bundles. The paired source, failure text and evaluation labels in this check are synthetic fixtures.

## Scientific controls

Phase 3 retained the exact Phase 2.5 fixture schedule, source graph, embedding bundle and shared seed and adjacency. The saved Phase 3 plan independently matched all eight earlier request payloads, evidence IDs, selection and rendered-evidence hashes, input counts and packing allowance. The frozen preregistration and its schema are byte-identical. T0 and T1 differ only by propagation and its effect on selected evidence. Prompt rendering and packing are unchanged. Model and device settings, seed, temperature, JSON format, output cap, serial execution, and RRF fusion with `k=60` and original-order ties remain shared. Paths, repair, typed edge weighting, method Full escalation and routing remain disabled.

Each ready request now records a hash of its complete message list, separate system and user content byte lengths, its prompt counter values, output cap, context window and physical reservation. Every candidate comparison retains selected IDs, source spans, scores, overlap, a signed input-size difference and the IDs rejected by the packer for budget. An infeasible mandatory package blocks both arms for that defect. A known-usage invalid verdict stays charged and excludes the pair from paired accuracy. When both arms were fully attempted, their usage differences remain available even if a verdict was invalid.

## Counting units

| Measure | Interpretation in this phase |
|---|---|
| Embedding tokenizer | BGE-M3 input counts from the saved encoder provenance. The longest of the eight inputs was 45 tokens for Java and 36 for Python. These are encoder inputs. |
| Byte proxy | UTF-8 bytes of the complete system and user message contents plus 256. The fixture allowance was 2,048 proxy units per candidate. |
| Local chat-template tokens | Exact token IDs for the explicitly supplied local template, including system and user roles and the assistant generation prefix. A small local tokenizer fixture passed against an independent count. |
| Serving-template tokens | Unverified and stored as `null`. The pinned local Ollama model and template layer exist, but the available loose tokenizer lacks a chat template, and serving parity was outside the offline check. |
| Provider-reported usage | Input and output counters returned with a completed reply. Phase 3 has synthetic provider counters only. They remain separate from planning estimates and reservations. |
| Conservative admission | The existing context check uses a heuristic estimate that includes the output cap. Each fresh call reserves `num_ctx + num_predict` for physical admission. A reservation is released into settled usage when a reply is reconciled. |

The exact-local counter now requests a flat sequence of token IDs and rejects unexpected output shapes. Its identity includes the loaded template. This also covers templates stored separately from the tokenizer configuration. The complete request is recounted against the packing count before it can be admitted. Local template counts carry no claim of serving parity.

## Reservation change

The previous fixed planner summed one full context-plus-output hold for every scheduled request. Because the collector is locked and serial, the physical exposure for request \(i\) is bounded by the previous settled usage plus that request's full hold. For a planned group in execution order, the new bound is

\[
\max_i\left(\sum_{j<i}\min(L_j,H_j)+H_i\right),
\]

where \(L_j\) is the existing logical reservation enforced after a completed call and \(H_j\) is the full physical hold. The planner applies the bound independently to each arm, defect and run, retaining the older sum for audit. The journal validates that paid requests form a prefix of the schedule and that settled usage is within both bounds before resume can add a call. If a request is interrupted before a trustworthy response, its full hold remains and execution stops. The Store still enforces its original per-call physical admission and its run and defect caps.

| Fixture scope | Sum of full holds | Serial bound |
|---|---:|---:|
| Java T0 / T1 | 8,512 / 8,512 | 6,171 / 6,357 |
| Python T0 / T1 | 8,512 / 8,512 | 6,104 / 6,385 |
| Java pair | 17,024 | 10,184 |
| Python pair | 17,024 | 10,078 |
| Whole synthetic schedule | 34,048 | 17,953 |
| Whole real-backend dry plan, same fixture | 34,048 | 34,048 |

The old calculation rejected the fixture limits of 6,385 per arm, 10,184 per defect and 17,953 per run. The new check admitted the same complete schedule at those limits and all eight synthetic requests settled successfully. Focused tests also rejected limits one token below the corresponding bound. No live reservation was reduced. Without serving parity, the real-backend logical reservation is the full hold, so the formula yields the original sum. For example, a 24-target arm at the draft 8,192 context and 160 output cap would still reserve 200,448 against a 36,000 cap. The preregistration's 35,584 last-request bound assumes a proven 1,184-token settled bound per preceding call. That proof remains open, and the planner does not use the hypothetical bound.

## Paired results

| Target | T0 / T1 complete byte-proxy units | T1 − T0 |
|---|---:|---:|
| Java `calculate` | 1,755 / 1,941 | +186 |
| Java `helper` | 1,752 / 1,787 | +35 |
| Python `calculate` | 1,688 / 1,969 | +281 |
| Python `helper` | 1,685 / 1,966 | +281 |

| Fixture defect | Synthetic input T0 / T1 | Synthetic output T0 / T1 | Synthetic input + output delta |
|---|---:|---:|---:|
| Java | 749 / 805 | 54 / 54 | +56 |
| Python | 716 / 857 | 54 / 54 | +141 |

Both arms had the same 2,048-unit allowance and 160-token output cap. The proxy and synthetic provider counters each show unequal input sizes. The run settled 3,343 simulated tokens and zero real physical tokens. Journal reuse counts toward an arm's logical usage and retains the original charge once in the simulated physical total. A completed resume issued zero additional requests. An interrupted run with one settled call reused that call and issued only the remaining seven. Malformed paid messages were retained verbatim in the raw ledger and did not yield a valid observation. Pending, unknown, out-of-order, or over-bound journal entries block resume.

The results preserve per-arm rankings, counters, calls and method/file metrics. Signed `T1_minus_T0` differences are stored per defect for logical and physical usage, MRR, Top-1 and Top-5. Accuracy deltas are populated only for joint-complete labeled pairs. The synthetic label scores exercise these fields for later cost–accuracy or Pareto analysis. They are fixture data.

## Checks and remaining gates

| Check | Status |
|---|---|
| Focused tests | **PASS**, 53 tests in 9.00 s. Coverage includes local template roles and prefix, UTF-8, duplicate evidence, empty/short/long source, reservation boundaries, paid invalid messages, and interrupted/resumed accounting. |
| Regression suite | **PASS**, 237 tests in 41.25 s, including the focused tests. Only an in-process HTTP fixture was allowed. |
| Plan reproducibility | **PASS**, independent Python hash seeds 7 and 19 produced byte-identical dry plans. |
| Linux portability | **PASS** for offline selector and admission parity. Network-disabled Linux aarch64 with Python 3.12.16 matched all 16 arm/target audits and both reservation envelopes across the fixture-mode and real-backend dry plans. |
| Real Defects4J inputs | **PENDING**. The configured smoke manifest is missing. Lang-54, Math-80 and Closure-61 still need official buggy checkouts, observed failures, compatible candidates and independently verified labels before a real run. |
| Serving-template parity and Linux/A100 execution | **PENDING**. Verify the pinned serving build, template/tokenizer, provider-reported usage and deployment limits before applying a tighter live bound. Linux re-encoding and real GPU execution remain unverified. |

Phase 4 is proposed as the synthetic end-to-end rehearsal and reporting prototype. It can exercise the saved paired results and failure cases, and validate the accuracy-versus-cost summaries and figures, after approval.

## Artifacts and reproduction

`results/graft_fl_phase3_fixture/` contains the two offline plans, their summary, Linux audit, and the synthetic run's immutable manifest, raw request ledger, SQLite journal, accounting snapshot and paired results. The [Phase 2.5 fixture](phase025.md) supplies the graphs and real embedding bundles. Run from the repository root with the project interpreter. Use fresh outputs and update saved resource paths in the JSON configuration if moving hosts.

```bash
graft_phase3_results=results/graft_fl_phase3_fixture
graft_phase3_output=$(mktemp -d)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=7 python -B \
  -m rave_fl.graft.fixed_evidence_experiment dry-run \
  --config results/graft_fl_phase25_fixture/config.json \
  --output "$graft_phase3_output/plan.json"
cmp "$graft_phase3_results/real-embedding-plan.json" "$graft_phase3_output/plan.json"
python -B -m rave_fl.graft.fixed_evidence_experiment run \
  --config "$graft_phase3_results/tight-synthetic-config.json" --output "$graft_phase3_output/run"
python -B -m rave_fl.graft.fixed_evidence_experiment resume \
  --config "$graft_phase3_results/tight-synthetic-config.json" --output "$graft_phase3_output/run"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m pytest -q \
  tests/test_graft_phase3_admission.py tests/test_graft_topology.py \
  tests/test_graft_phase2_integrity.py tests/test_graft_fixed_evidence_*.py
```

The real-backend diagnostic uses `real-backend-dry-config.json` with `dry-run`. Phase 3 keeps real execution disabled in the fixed runner. Existing manifests remain readable as archived data. Resume compares the implementation identity, so the revised runner rejects earlier execution manifests instead of appending to them.
