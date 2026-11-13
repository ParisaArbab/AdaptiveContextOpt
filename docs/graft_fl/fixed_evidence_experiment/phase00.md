# Phase 0 · fixed-evidence experiment audit

## Scope and baseline

Local branch `GRAFT-FL`, HEAD `e64d6dee9e642adaf08e4485512e7060f3335403`, reviewed 12 November 2026. The local branch ref matches the revision in the supplied master prompt. The user explicitly approved proceeding with the existing unrelated untracked files. The initial tracked tree was clean. [Draft preregistration and SHA-256 inventory](preregistration.json) · [Schema](preregistration.schema.json)

## Current code and the Phase 1 entry point

| Step | Reusable code | What the fixed comparison needs |
|---|---|---|
| Resolve benchmark inputs | `eval.adapters.load_instances`, `eval.runner.validate_checkout` and `prepare_one`, `eval.candidates.provide` | Prepare once. Freeze the ordered entities, candidate identities, prior ranks, source spans, evidence and candidate pool hashes. The existing checkout validator checks the declared revision and tracked cleanliness. It needs independently supplied official revision provenance. |
| Build one evidence graph | `graft.graph.build_graph`, `Graph.hash` | Reuse one graph for both arms. It builds bounded Java/Python method evidence with source origins. `MAY_CALL` is a static possibility, `REQUIRES` is a lexical dependency proxy, and coverage requires observed `CoverageFact` input. Flag parse fallbacks and ambiguous frame mappings before accepting prepared inputs. |
| Encode exact texts | `graft.selection.query_text`, `graft.encode.export`, `graft.topology.EmbeddingBundle.load/lookup` | Export once for graph-node text and the failure-plus-target queries. Freeze graph, bundle, encoder and exact text hashes. Validate genuine weights and provenance, complete vectors and absence of truncation. |
| Select and pack | `graft.topology.topo_select`, `propagate`, `graft.selection.pack_scores/package/render`, `TokenizerCounter` | Supply frozen `TopologyConfig` values to `topo_select`. Both variants construct dense/sparse scores, acceleration and neighborhoods. T0 passes the seed scores to the packer and T1 passes the propagated scores. Packing uses score minus redundancy penalty, so a score-per-token policy would change the comparison. Render and count the complete messages in both arms. |
| Admit and record each fixed action | `graft.controller.Session.planning/execute`, `GraphBackend.configure/payload/request_identity/execute`, `eval.backend.Backend.planning/execute`, `Store.request` | Use one Session per arm and call `Session.execute([candidate], EvidenceStage.GRAPH, selection)` in the frozen order. GraphBackend checks source, target, failure text and package identity, and uses the single-verdict parser. Store handles exact cache identity, physical reservations and paid or unresolved requests. |
| Rank and evaluate | `eval.fusion.Fusion.scores/order`, `CandidateState`, `eval.metrics.load_gold`, `graft.evaluation.evaluate_profile` | Keep one graph observation per target, using the same observation representation and RRF in both arms. Read evaluator-only labels after collection. The existing strict Java dispatch is specific to Defects4J. Other datasets need demonstrated method-level contracts and a suitable evaluator before use in the primary comparison. |
| Record runs and summarize | `eval.runner.implementation_identity/run_lock`, `graft.runner.resource_identity`, `graft.reporting.report` | Reuse manifest, locking and accounting primitives. The existing reporter derives expected arms from legacy policy names and matches all configured comparisons, so the new report needs an explicit T0/T1 match set that stays independent of optional controls. |

The smallest injection point is **below Router and above Session**. The proposed opt-in runner supplies prepared selections to the existing action transport. Current `graft.runner.run` instantiates one `GraftConfig` shared across policies. Current `Router.execute` starts with L1 and filters available graph actions by packing, remaining budget and prior stage. `l1_graph` chooses the *eligible* target with the smallest prior rank number subject to caps. `graft_adaptive` uses changing scores, uncertainty and costs and can escalate to Full. Either path can change inspected targets between selectors. Current `select --selector topo --no-propagation` handles a single target. The paired runner will reuse the selector and transport directly.

## Preregistered comparison

T0 `dense_sparse_no_propagation` and T1 `dense_sparse_with_propagation` use the same candidate pool, prior, ordered targets, source revision, failure evidence, graph, query, embedding bundle and renderer. For each bug the schedule is every candidate in the frozen prepared pool exactly once, ascending prior rank with prepared-order tie resolution. The draft cap is 24 per bug and applies during initial candidate preparation. A failed target stays in the schedule and makes the pair incomplete. Identical settings include dense mixture 0.5, acceleration 0.2, semantic and sequential weights 1.0, redundancy 0.2, five neighbors, restart factor 0.85, at most 300 propagation iterations and tolerance 1e-9. Program edges are disabled for propagation. Paths and dependency repair are both disabled. The common renderer can still include typed links between selected pieces. That is a property of the shared rendering of the selected evidence.

The only changed selector flag is `TopologyConfig.propagate`. The output evidence may differ in content, links and length under the common allowance. Record both selections and exact messages, input units, rejection reasons, graph and bundle hashes, and selected-node overlap. Compare which pieces were promoted against T0 after collection. A change in package size is an observed consequence of the selector and must be reported.

The common graph prompt is `graft-graph-verdict-v2-short-strings`. Pin its system hash, renderer, provider, model digest, quantization, context, output limit, seed and temperature. RRF means Reciprocal Rank Fusion, with `k=60`, original-prior tie breaking and no rounding. The draft Mac condition pins the installed `llama3.1:8b` digest and Q4_K_M, seed 0, temperature 0, context 8,192 and 160 output tokens per inspection. Provider version, encoder revision and tokenizer equivalence to the served template remain pending. A different model or serving condition gets a separate condition and manifest.

The proposed input allowance is **1,024 chat-template tokens for the entire system and user messages**, including the generation prefix and common rendering. It is a planning choice to cap the pilot. The current `ByteCounter` is UTF-8 content bytes plus 256. If a local tokenizer cannot be verified against the serving template, a separately declared proxy comparison can use the same byte allowance in both arms and report proxy units alongside provider counters. Its allowance is expressed in bytes. Use strict provider admission and preflight both arms before any paid request. If either mandatory package or the complete planned sequence is infeasible, record both arms as pair-incomplete and preserve the individual reasons. Changes to allowance or model settings require a new recorded condition before outcomes are inspected.

Preflight must also address the following source-level risks.

- `TokenizerCounter` measures its local template. Its `exact` flag alone cannot establish parity with the served model. Freeze vocabulary and full template identity and check the provider contract. The backend also applies its own conservative prompt estimate and reserves context plus output for physical admission. Check both against the planned messages.
- `EmbeddingBundle` checks vector shape, dimensions, finiteness and exact text lookup. Its metadata does not by itself prove the claimed checkpoint produced the vectors. Bind a locally verified encoder revision and file hashes to the prepared bundle and reuse its exact bytes in both arms. A synthetic bundle can exercise offline tests only.
- `GraphBackend.request_identity` already includes complete messages, provider/model settings, revision, candidate, graph, selection and configuration identity. Preserve that key and scope the initial smoke to its own Store/cache. An exact recorded observation may be reused on resume, with logical usage retained. Changed payloads or identities require distinct judgments. Freeze the scheduled targets before any response or cache cost becomes visible.
- `GraphBackend.execute` checks numeric bounds via the shared parser and requires uncertainty and string fields for graph verdicts. The current parser does not enforce all object keys or the requested explanation length. Validate the same declared schema in both arms, retain paid invalid replies, and record output-limit stops. Invalid or missing responses make the pair incomplete for matched accuracy. Unknown usage retains its reservation and stops further paid collection. Preserve the individual execution status even when the paired status is incomplete.
- The evaluator needs exact overloaded signatures resolved against the buggy revision. `java_strict_v2` is case-sensitive and handles explicit parameters, but it accepts simple versus qualified type names by matching the suffix and erases generic arguments. Audit ambiguous overloads before admitting verified labels. Preserve label counts and reasons for unavailability.
- Keep gold in an evaluator-only input keyed by defect, separate from the online manifest and runner resources. Reuse `load_gold` after collection with the evaluator's source descriptor. Test that preparation and the fixed collector cannot reach the label loader, including on resume. Freeze the label artifact identity for evaluation without passing gold values to the collector.

### Outcomes and split policy

The primary statistic is the mean of per-bug T1 minus T0 MRR on the jointly complete, strictly labeled pairs. MRR is Mean Reciprocal Rank and uses the position of the first correctly identified method. Keep known labeled bugs whose candidate pool misses the gold location in the denominator with zero rank credit. In the operational view, a known-labeled incomplete arm contributes zero credit. Preserve actual completed status by arm and report operational results separately from matched pairs. Missing or unresolved labels are unavailable. Report an additional any-gold-in-pool diagnostic with its own count.

Secondary outcomes are Top-1, Top-3, Top-5, candidate-pool recall, per-bug wins/ties/losses, completion, invalid responses, logical and physical tokens, reused counters, unresolved reservations and per-target evidence usage. Report graph build, selection, embedding time/device/peak memory and wall time separately from verification tokens. Charge one shared encoding pass physically and show the shared preprocessing separately when comparing standalone policy cost. Record retrieval and test preparation as upstream work.

The three named bugs are feasibility cases. Use descriptive per-bug results for one to three cases. After a successful and separately approved smoke, freeze 10–15 independent exploratory defects across at least four projects before seeing results. Use official defect IDs/revisions, exclude known inspected, pilot and smoke cases, and check external experiment registers. Fix the split and input fingerprints before collection. Reserve a separate untouched held-out set for confirmation after choices are frozen. Report the exploratory paired differences and number of projects. The draft calls for descriptive reporting at this size, and a project-cluster bootstrap only on a later cohort with adequate clusters and coverage. As a conservative planning threshold, use at least 30 matched defects and eight projects before attempting a 95% project-cluster interval with 10,000 fixed-seed resamples. Retain descriptive results if clusters are too sparse or dominated by one project. This threshold is a reporting choice and cannot establish power.

A positive exploratory signal requires improved paired MRR with reproducible evidence-retention examples, documented overhead and no substantial operational losses. Check the prior and separately labeled Full reference before proposing a quality claim. A flat or adverse paired result, or an apparent gain that requires a changed schedule, fusion or allowance, is a reason to report the result and reconsider the experiment.

### Minimum-cost sequence and draft caps

Start with offline preparation and a dry-run at **zero provider calls and zero new inference tokens**. Include prior-only for zero additional verification calls, with retrieval cost separate. For a later approved local feasibility run, start with one eligible preselected bug and only T0/T1. Extend to the remaining two only within the approved phase and caps. Make real BGE preparation an explicit offline operation. Current Phase 0 has no approved paid execution.

| Proposed T0/T1 smoke scope | Logical token cap | Physical token cap | Fresh call cap |
|---|---:|---:|---:|
| One arm on one bug | 36,000 | 36,000 | 24 |
| T0 and T1 on one bug | 72,000 across both arms | 72,000 | 48 |
| Up to three bugs | 216,000 across all arms | 216,000 | 144 |

These are ceilings, without an expected spend estimate. With verified 1,024-input and 160-output bounds and the current 8,352-token physical reservation, the arithmetic for admission of the last request in a 24-target arm is 23 × 1,184 + 8,352 = 35,584. This depends on template equivalence and actual provider counters. Recompute admission for the actual frozen schedule. Give each arm its own envelope and precheck the combined physical cap. Current Store enforces run and bug caps. The new collector must enforce a per-arm fresh token/call envelope from the journal before handing over an action. Use a single locked serial run and preserve spending across resume.

Add contextual controls only in separately budgeted runs. B is BM25 with the same renderer, anchors and target schedule as T0/T1. L is the existing Fixed L1 policy with its native prompt. Full can have a 36K equal-budget attempt with completed target coverage reported. If incomplete, an optional **60K per bug and arm** Full reference uses a distinct manifest, up to 24 fresh inspections per bug, and reports any remaining incomplete coverage. The Full and L1 comparisons do not isolate propagation. Keep typed propagation T2 for a later approved secondary contrast. A larger cohort or additional arms require a new preflight and their own approved caps.

## Historical benchmarks and live readiness

Historical Qwen2.5 7B development collection had 97 model attempts, 93 jointly completed cases and 84 jointly completed cases with labels. The 8,192 setting is per-request context. The historical policy allowance was 120,000. Table rows below all use the matched 84. Frozen RRF reorders stored judgments and holds purchased actions fixed. [Metric table](../../../reports/graft_fl_development_2026_11_05/tables/frozen_ranking_metrics.md)

| Matcher and ranking condition | Prior MRR | Full MRR | Fixed L1 MRR | V2 MRR |
|---|---:|---:|---:|---:|
| Historical matcher and historical ranking | 0.5282 | 0.4997 | 0.3981 | 0.4035 |
| Historical matcher with frozen RRF | 0.5282 | 0.5521 | 0.4907 | 0.5049 |
| Strict Java matcher and historical ranking | 0.5178 | 0.4856 | 0.3868 | 0.3932 |
| Strict Java matcher with frozen RRF | 0.5178 | 0.5395 | 0.4795 | 0.4937 |

The two-bug Llama3.1 8B Mac smoke used existing trigger descriptions on Lang-1 and Time-1. Revised arms completed with GRAFT and Fixed L1 at ranks 2 and 1. Adaptive GRAFT used 11.55% more verification tokens on average. This records local compatibility with the existing router and no observed ranking gain. A paired propagation-only result is **NOT MEASURED**. [Local smoke report](../../../reports/graft_fl_local_real_smoke_2026_11_05/README.md)

Local host is arm64 macOS 26.1, with 18 GiB RAM and 51 GiB available disk at inspection. The localhost Ollama tags endpoint returned Llama3.1 8B Q4_K_M with the pinned digest, plus Qwen3-VL 8B and two DeepSeek-R1 tags. This was a metadata read. The default Python 3.11.9 lacks project dependencies. The installed pyenv 3.11.0 imported all 13 required modules and has PyTorch, Transformers and parsers. `FlagEmbedding`, a local BGE-M3 checkpoint and a provider-equivalent tokenizer were not found in the checked locations. The loose tokenizer in Downloads lacks a chat template and is not established as belonging to the pinned model. Defects4J and NVIDIA tools were not on PATH. Actual A100 service, hardware and provider behavior remain unverified.

| Preselected bug | Official checkout | Observed failure evidence | Matching ranker files | Labels | Previous use |
|---|---|---|---|---|---|
| Lang-54 | Unavailable locally | Unavailable | Unavailable | One FlexFL entry, source verification pending | Absent from inspected run catalog and local smoke |
| Math-80 | Unavailable locally | Unavailable | Unavailable | One FlexFL entry, source verification pending | Absent from inspected run catalog and local smoke |
| Closure-61 | Unavailable locally | Unavailable | Unavailable | One FlexFL entry, source verification pending | Absent from inspected run catalog and local smoke |

`data/graft_smoke/instances.json` is missing. All three live cases are **BLOCKED** pending official buggy revision and clean checkout, failing test commands/log hashes and toolchain, source-aligned FlexFL CSVs and strict label verification. The archived exclusion check does not cover external experiment registers. Keep the specified cohort and record unavailable cases. The old Time-1 and Lang-1 checkouts remain development inputs.

## Phase 1 proposal pending approval

- Add `rave_fl/graft/fixed_evidence_experiment.py` for preparation, named paired configurations, schedule serialization and the thin Session loop. Use shared runners for data loading and resource identity. Add an opt-in entry point with a provider-free dry-run. Keep CLI and Router interfaces intact.
- Add `tests/test_graft_fixed_evidence_schedule.py` and `tests/test_graft_fixed_evidence_accounting.py` for same candidates and byte-stable plans, the one-flag diff, deterministic recorded schedules, completeness, sealed labels, per-arm caps and paid/unknown/cache cases. Use fake inputs and the existing regression suites. Include a check that the fixed collector never calls Router.
- Add `configs/graft_fixed_evidence_dry_run.json` with explicit fake/provider-free settings. Record subsequent validation in this phase directory. The portable launcher, real embedding work and live smoke follow their own approved phases.

## Executed checks

| Command or invocation | Result |
|---|---|
| `git branch --show-current`, `git rev-parse HEAD`, `git show-ref --verify refs/heads/GRAFT-FL`, `git diff --exit-code HEAD`, and `git status --porcelain=v1 -uall` | PASS for branch, revision and tracked cleanliness before notes. Existing untracked files accepted by the user. |
| `graphify query "GRAFT-FL fixed schedule GraphBackend Session topo_select propagate prepare_one Store Fusion" --budget 1100` | Exit 0 with a package/skill version warning. Index surfaced legacy sketch nodes. The source map above comes from direct inspection of current source. |
| `$HOME/.pyenv/versions/3.11.0/bin/python -B -m rave_fl.graft.cli --help` and `... -m rave_fl.graft.cli select --help` | PASS, exit 0. Current entry points and selector toggle present. |
| `$HOME/.pyenv/versions/3.11.0/bin/python -B -` with `importlib.import_module` on `graft.{graph,selection,topology,encode,controller,runner,evaluation,reporting,cli}` and `eval.{fusion,backend,runner,metrics}` under `rave_fl` | PASS, exit 0, 13 imports. Optional FlagEmbedding unavailable. |
| `uname -sm`, `sw_vers`, `sysctl -n hw.memsize`, `df -h .`, `command -v ollama nvidia-smi defects4j`, filesystem/cache inspection, and a 2-second `GET http://127.0.0.1:11434/api/tags` | PASS for local inventory and provider metadata. The external-resource gaps are recorded above. |
| Python read-only checks of the archive catalog, three proposed IDs, local checkout paths, FlexFL data and historical reports | PASS for baseline presence and audit evidence. 97 main manifests and 97 corresponding results files present. All three new live inputs blocked. |
| Draft 2020-12 schema and additional invariant validation documented in [PHASE_LOG.md](PHASE_LOG.md) | PASS, including exact configuration diff, source fingerprints and negative controls. |
| Regression suite, live model, encoder execution, downloads, official test runs and server checks | SKIPPED in Phase 0. Model calls 0, generated tokens 0, provider verification tokens 0. |

Phase 0 design gate passes. Live resources are tracked as later execution blockers.
