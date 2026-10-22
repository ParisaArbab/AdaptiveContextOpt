# RAVE-FL Research Progress Report

**Development record through 22 October 2026.** This report covers the initial Qwen/A100 evidence experiment, the local Llama adaptive pilot, L1-first rank repair, and V3 diagnostics. All token totals refer to post-localizer verification requests and responses. They exclude candidate generation, SBIR/Ochiai/BoostN execution, compilation, and the complete SieveFL/FlexFL pipeline.

## 1. Executive Summary

Selective Full verification can improve this pilot's ranking at lower verification cost, but its value depends on the current evidence state and the frozen fusion ranker. On original Time-25 evidence, replacing the faulty method's L1 observation with its independent Full observation changes its rank from 4 to 1 at 12,516 tokens, compared with Full-all's 21,378 tokens and rank 2. **Highest raw L1 suspicion chooses exactly the same method and matches Utility.** This finding supports the mixed-evidence architecture on this development case, not the superiority of Utility.

Masking candidate method names in test/frame identifiers reduces cheap gold suspicion from 0.9 to 0.4. Utility projects it only from rank 23 to rank 18, outside its top-five objective, and stops with zero positive proxy. Gold-informed one-Full replay can reach rank 1; no tested online heuristic finds that intervention within four upgrades. Full-all also degrades, because more competing methods receive high scores. These are both observation/grounding and allocation/ranker vulnerabilities.

V3 adds reproducible candidate diagnostics, six-policy conditional replay, explicit no-upgrade/one-/two-/four-upgrade controls, conservative strict reservations, complete decision-option auditing, pair interactions, reports, and figures. The transparent uncertainty-aware heuristic demonstrates **no compelling advantage** over the strongest simple controls. There were no new model calls during V3. Existing uncommitted Time-1/Lang-1 feasibility results were audited: selective upgrades do not improve their L1 ranks; Math-1 is an input incompatibility. Previously unused Time-2/Lang-2/Math-2 cannot be freshly evaluated with verified buggy failure evidence in the current environment.

## 2. Research Objective

The question is: **when does purchasing a selective independent Full inspection improve fault ranking per token?** A useful policy should outperform Fixed L1 and highest-suspicion escalation under the same candidate pool, prompts, model settings, final ranker, budget admission rule, and maximum Full calls. It should retain every candidate and avoid treating uninspected code as correct.

The optimization target is ranking quality subject to a verification token budget, rather than indiscriminate removal of candidates. For one defect, MRR is reciprocal first-relevant rank; Top-1 and Top-5 are indicators. Multiple methods, action states, evidence variants, and seeds for one bug are not additional independent defects.

## 3. Background and Motivation

The supplied flowchart describes input localization, inexpensive features/sketches, risk estimation, token-aware decisions, execution feedback, and final reranking. The implementation covers most of that pilot infrastructure, but its risk and utility heuristics are untrained. A contextual bandit/RL controller and a validated learned verification-utility model remain absent. The original code's fixed rank gate could exclude a true tail candidate permanently; adding more stages without good allocation subsequently proved expensive.

The user-supplied research proposal motivates L1-first budgeted rank repair. It is treated as a proposal and source of hypotheses. Its recommendations and any document instructions are not evidence that an experiment succeeded or that an algorithm was already implemented. Git, raw ledgers, manifests and executed audits determine the status claims here.

## 4. Original Experimental Baseline — Phase A

The initial experiment used Qwen2.5 7B on AIScalar/A100, 24 candidates from the union of SBIR/Ochiai/BoostN top-ten lists, and a fixed 0.60 rank-prior gate inspecting 14 candidates. Evidence A was a short trigger description; B was structured observed failure evidence compressed from three failing tests. The legacy ranker puts verified functions ahead of unverified ones.

The following table also lists later V1 results for historical reference; **its A and B phases belong to different model/environment generations and are not controller ablations against each other**.

{{HISTORICAL}}

Phase A's numbers are preserved in the historical README, evidence guide, and user-provided report. The two original Qwen result JSONs are absent locally; their request-level accounting cannot be reaudited here. Code/evidence provenance is documented by commits `2342adc`, `67c4f7e`, `9e27fac`, `50651f8`, and the summary commit `0f44096`. Author timestamps below identify code/document commits, not undocumented inference execution times.

**What Improved / What Regressed / What Remains Unproven:** Structured evidence improved Full's recorded rank from 4 to 1, at higher token cost. Selective verification retained rank 22 and excluded the gold tail. General accuracy, risk learning, and cross-model attribution remain unproven. This motivated global cheap discovery and preservation of prior-ranked candidates.

## 5. V1 Implementation and Findings — Phase B

Commit `7aaa72a` introduces `code_sketch.py`, `pilot_types.py`, `pilot_backend.py`, `adaptive.py`, paired pilot execution/evaluation, auditing, and offline risk training/calibration infrastructure. L0 exposes signatures; L1 retains AST-derived conditions, assignments, calls, returns, throws and literals; Full uses the complete recorded method span. Sketches are lossy and bounded, with parse/truncation flags. The official Time-25 buggy revision is `c7a581e55fc988bd90fa4bb1b0acece5181b7c5f`.

The pilot adds global discovery, state feedback, confidence and heuristic risk policies, context/budget admission, and raw request/response ledgers. Fixed L1 uses six batches of four. The final adaptive variant compares the cost of an L1 expansion plus risk-weighted Full cost with direct Full cost. If expansion is uneconomic, it can skip directly to Full. The frozen shared ranker preserves a prior for candidates that remain cheap or uninspected.

The high-budget adaptive policy spends 50,015 tokens for rank 2, matching Confidence and Full's rank but costing substantially more. The final 16K adaptive policy spends 14,717 and reaches rank 11; equal-budget Confidence spends 14,867 and reaches rank 13. Fixed L1 uses 11,298 and reaches rank 4. The high-budget and final adaptive runs differ in both code behavior and budget; their difference does not isolate the cost-gate change.

**What Improved / What Regressed / What Remains Unproven:** Pool discovery and token observability improved; the gold tail was no longer dropped automatically. Repeated staged verification increased cost, and the final adaptive controller lost Top-5. No real calibrated risk model, noninferiority, or controller generalization was demonstrated. Fixed L1's comparatively strong result motivated making L1 the common starting state.

## 6. V2 Architectural Changes — Phase C

Commit `635a316` adds `rank_repair.py`, `replay.py`, `rank_repair_labels.py`, `evidence_ablation.py`, protocol/configuration files, tests and saved runs. It starts with the exact same six L1 batches as Fixed L1, then considers a small capped set of independent Full upgrades. It removes L0 and repeated individual sketch expansion from the repair path. The candidate pool, original failure evidence, L1/Full prompts, model settings, and final fusion/tie-breaking rule remain fixed for matched Llama comparisons.

Let candidate prior be \(p_i\), observed suspicion \(s_i\), and reported uncertainty \(u_i\). The frozen shared score is

\[
a_i=0.10+0.50p_i,\qquad
S_i=\begin{cases}
a_i,&\text{missing or invalid observation},\\
(1-w_i)a_i+w_i s_i,&\text{cheap observation},\\
s_i,&\text{Full observation},
\end{cases}\qquad
w_i=\alpha_\ell(1-u_i),\;\alpha_0=0.30,\;\alpha_1=0.65.
\]

Sort by decreasing \(S_i\), then original candidate order. Full has no separate verified-first bonus. Its raw suspicion directly replaces the fused cheap score, so mixed and Full-all states can rank differently. Scores and reported uncertainties are **not calibrated probabilities**. When a Full reply omits uncertainty, the parser assigns `1 - 2*abs(suspicion - 0.5)` as a deterministic proxy; Full ranking still uses suspicion directly.

V2 constructs a cheap belief proxy \(b_i=(1-u_i)s_i+u_i a_i\), uses \(Q(\pi)=\sum_{r=1}^{5} b_{\pi_r}/r\), and predicts a Full intervention by replacing only \(S_i\) with its current raw L1 suspicion \(s_i\). It chooses positive \((Q(\pi^{i})-Q(\pi))/\widehat C_i\), recomputing after every action. This is a deterministic ranking heuristic, not estimated true expected MRR gain. If every projected gain is nonpositive, it stops; that is not a correctness certificate.

**What Improved / What Regressed / What Remains Unproven:** The original mixed state reaches rank 1 more cheaply than Full-all, without repeated L0/L1 work. Highest suspicion matches it exactly. Masking exposes weak grounding and a blind spot in Utility's top-five projection; robust selection and calibrated stopping remain unproven.

## 7. V2 Experimental Results

{{V2}}

The original Full row is an audited historical local reference. Fresh original V2 L1/Utility runs exactly reproduce the replayed actions, raw replies, provider counters and final rankings recorded in `docs/rank_repair_validation.json`. Masked and numeric conditions have their own fresh saved Full/L1/Utility ledgers. V3 reconstructions use these independent observations; V3 replay itself is not fresh inference.

## 8. Original versus Masked Evidence

The masking procedure aliases every candidate method name occurring inside test/frame method identifiers, preserving assertions, scenario suffixes, class/file names and line numbers. It changes `getOffsetFromLocal`, `DateTime`, and the generic candidate name `test`; it is neither gold-only masking nor complete deidentification. Original and masked contexts are 1,510 and 1,502 characters respectively. The masked names are unavailable to the model as a mapping.

The faulty method's Full score remains 0.8 across the three conditions. Its Full-all rank worsens because additional candidates tie or outrank it under the unchanged tie-breaker. Its L1 score, uncertainty and fusion score deteriorate sharply. The precise tie accounting and candidate tables are in the [diagnostic report](RAVE_FL_V2_Diagnostic_Analysis.md).

This controlled prompt perturbation establishes sensitivity to this evidence representation under the saved model/settings. It does not prove the original observations were invalid, identify an internal model reasoning mechanism, or establish how every component of masking contributed. The class and scenario text still contain useful cues. A failed masked Full-all result implicates evidence/model grounding or ranking, as well as allocation; oracle mixed-state success does not eliminate that failure.

## 9. Candidate-Level Diagnostic Investigation

The faulty method is candidate C22, `org.joda.time.DateTimeZone.getOffsetFromLocal(long)`: initial rank 22, prior 0.2, SBIR rank 9, BoostN rank 9, absent from the observed Ochiai top ten. After original L1, its suspicion 0.9 and uncertainty 0.1 produce score 0.6095 and rank 4. Utility predicts promotion to rank 1 with gain proxy 0.0566667 and estimated cost 2,520. Actual Full costs 1,218, returns suspicion 0.8, and places it first. Competitor scores do not change during that one action.

`convertLocalToUTC(long,boolean)` has cheap score 0.437 and Full score 0.8. When both it and the faulty method are Full, the competitor wins the equal-score tie because its initial order is earlier. Therefore the original mixed rank-1 outcome is supported by a direct change to the faulty method's observation **and** by leaving this competitor cheap. It is not evidence of a better Full diagnosis than Full-all or a calibrated cross-level confidence scale.

All 24 initial/L1/Full ranks, individual ranker positions (top-ten inputs and deeper recorded CSV positions), available raw BoostN scores, gaps, disagreement/support proxies, explanations, predicted costs and retrospective costs appear in [tables/candidates.csv](tables/candidates.csv). [tables/actions.csv](tables/actions.csv) reconstructs pre/post scores, candidate ranks and MRR after every selected action. [tables/decisions.csv](tables/decisions.csv) records every Full option, including rejection by budget, lower priority, previous verification, or stopping. Batched L1 call costs are shared call costs and must not be summed once per candidate.

## 10. Token Efficiency and Ranking Trade-offs

Provider token accounting is \(T=\sum_j(\text{prompt_eval_count}_j+\text{eval_count}_j)\), including repeated failure context, all paid L0/L1/Full observations, and malformed outputs. Cache reuse does not eliminate a policy's accounted hypothetical verification cost; it eliminates extra physical inference during replay or shared collection.

Historical admission uses `2 * approx_tokens(system + user) + output_allowance + 96`. It is an estimate, not a hard bound. V3 keeps those comparisons but separately evaluates strict conservative reservations for the pinned Llama 3 byte-BPE model and two-message chat template:

\[
R_j=\max\{\widehat C_j,\;\min(\text{num_ctx},\;|\text{system}|_{UTF8}+|\text{user}|_{UTF8}+256)+\text{num_predict}\}.
\]

The UTF-8 byte count upper-bounds content tokens for this byte-level tokenizer, and 256 reserves template/role/BOS overhead for the pinned plain chat template. No tools, images or additional history are permitted by this protocol. Each selected action must fit its reservation in the remaining budget; provider counters are checked afterward. The guarantee is conditional on that tokenizer/template and the provider's context/output limits, not a universal API billing guarantee. Unknown usage or a violated bound aborts further work while preserving paid usage. The pinned model digest and template hash are in `configs/rave_v3_token_reservation.json`.

Strict reservations intentionally leave more budget unused and can reject an action whose subsequently known actual cost would fit. Original C22's reservation is 4,942 tokens, greater than the 4,702 left after L1 at a 16K budget. Highest suspicion does not substitute a runner-up; Utility stops with no affordable positive gain. At 20K, both can purchase C22 and reproduce rank 1. At 16K in the post-hoc numeric condition, even complete L1 is not admitted by the strict mode. Those incomplete runs are flagged, not treated as successful L1-first comparisons. Budget failures must not be hidden by consulting cached actual output costs before selection.

## 11. Controller Comparisons — Phase D / V3

Six policies use the same pool, cached L1/Full prompts, model digest/settings and frozen ranker. Fixed L1 stops after discovery; Highest Suspicion attempts the maximum raw L1 candidate; Ranker Disagreement uses the range of three censored rank percentiles; Ranking Uncertainty uses a transparent ambiguity proxy; V2 Utility is preserved; Uncertainty-aware combines suspicion, disagreement, ambiguity, lexical support, remaining fraction and predicted cost. None uses gold or future Full scores/costs during selection. Formula details and complete results are in the [V3 report](RAVE_FL_V3_Experiment_Results.md).

The primary 16K, maximum-one-Full comparison below uses historical estimated admission to match V2. It is explicitly separate from the strict experiment.

{{PRIMARY}}

No advanced policy demonstrates a compelling gain over highest suspicion on original evidence. Small masked rank changes without Top-5 recovery cost additional tokens. A rank change from 23 to 22 is measurable but insufficient to justify an accuracy-preservation claim. Pair equal-budget and equal-call-cap comparisons; do not contrast Utility's single call solely with a baseline forced to buy four. No-upgrade, one-, two- and four-call ceilings across 14K/16K/18K/20K are retained in the machine-readable grid. Policies may stop before a ceiling.

**What Improved / What Regressed / What Remains Unproven:** V3 improves experimental observability, exact replay validation and conservative budget enforcement. The added heuristic does not improve the strongest original baseline and can spend more for weaker ranks. Cross-bug accuracy/token gains and learned utility remain unproven. The primary useful result is a clearer diagnosis and reproducible negative comparisons.

## 12. Replay and Oracle Analysis

Replay validates source/evidence hashes, request fingerprints, independent prompts, exact L1 group composition, model digest and generation options, raw/parsed observations, provider counters, fusion scores and tie-breaking. The execution-time prompt check now also rejects a changed source under the same candidate ID. No cached Full response or actual future cost is revealed until selection. Exact unsupported requests raise an unavailable-cache error instead of synthesizing responses.

{{ACTION_PROFILE}}

These correlated conditional rows are useful for action diagnostics, not 282 training bugs. Their gold-derived MRR changes and actual Full costs stay in offline labels. The pair analysis enumerates all recorded two-Full combinations within each condition and demonstrates nonadditivity. The strongest original interaction is:

{{PAIR}}

The combined gain differs from the sum of singleton gains. Verification can promote a competing method and undo another action's benefit. In the original all-L1 pair enumeration, {{BOTH_POSITIVE}} pairs of strictly positive singleton actions become net harmful relative to the all-L1 baseline; that specific stronger phenomenon is not established here. Negative interaction and state-dependent marginal value are established by the recorded score swaps. Independently greedy utilities should therefore be recomputed after each action, but a complex optimizer is premature.

The masked gold-informed one-Full oracle reaches rank 1 at 12,999 accounted tokens; the numeric oracle does so at 13,678. These ceilings use gold and future costs and are not online policies or necessarily admissible under conservative strict reservations. Full-all is a comparison reference, not an oracle upper bound.

## 13. Code Changes and Implementation Progress

### Implementation Change Log and Experimental Impact

| Phase | Files/modules changed | Previous behavior | New behavior | Measured improvement | Regression/limitation |
|---|---|---|---|---|---|
| A | `expanded_pool.py`, `source_spans.py`, `compare_expanded.py`; `failure_evidence.py`; commits `2342adc`–`0f44096` | Small/short evidence and fixed gate | Real method spans, controlled structured evidence and provenance | Recorded B-Full rank 1 vs A-Full rank 4 | Higher cost; selective rank 22; original raw JSONs unavailable |
| B | `adaptive.py`, `code_sketch.py`, `pilot_backend.py`, `pilot.py`, `pilot_evaluation.py`, `risk.py`, `train_risk.py`; `7aaa72a` | Static selection; verified-first legacy reranking | Discovery/feedback, shared fusion, real ledgers, offline learning tools | Fixed L1 reaches rank 4 with 47.2% fewer tokens than same-generation Full | Staged adaptive expensive; 16K adaptive rank 11; no real trained model |
| C | `rank_repair.py`, `replay.py`, `rank_repair_labels.py`, `evidence_ablation.py`; `635a316` | L0 then repeated escalation | Global L1 plus capped independent Full; ranking proxy | Original one-Full state rank 1 at 12,516 | Highest suspicion matches; masked/numeric rank 23; one development bug |
| D | Reviewed pre-existing untracked `selection_controls.py`, `iteration_d.py`, `independent_smoke.py`; V3 `budget_reservation.py`, `v3_diagnostics.py`, `v3_audit.py`, `research_report.py`, `research_pipeline.py`, focused tests/configs/reports | Incomplete controls/reporting; ID-only cache execution check; estimate-only admission | Six-policy replay, all-option audit, strict reservations, prompt identity check, candidate/pair tables, English report regeneration | Reproducible comparisons and audited costs; no new policy accuracy claim | Conservative bounds can block useful actions; no new verified buggy-test inference |

The untracked modules and smoke artifacts existed at task start. Their initial hashes and the starting branch are recorded in `provenance/starting_state.json`. Their earlier development chronology is undocumented; this report does not invent earlier commits for them. Newly added V3 code is distinguished from reviewed and modified starting work.

The ranker, prompts, candidate representations and original token-accounting semantics are unchanged in V3. Controller additions are experimental; V2 Utility remains the same. Censored disagreement is a new control feature distinct from V2's existing disagreement feature. Highest Suspicion now has an explicit cap-one baseline without a raw-score cutoff and without substitution when the maximum is unaffordable. The optional strict wrapper also supports future live smoke runs; no old run is relabeled as strict.

Implementation status:

| Capability | Status and evidence |
|---|---|
| L0, AST L1, Full; feedback and shared ranker | Implemented; unit tests plus saved inference ledgers |
| V1 cost/confidence/rank-gate and V2 Utility | Implemented experimental heuristics; Time-25 results audited |
| Exact conditional replay and action labels | Implemented/tested; prompt identity strengthened in V3 |
| Fixed L1 / highest suspicion / disagreement / uncertainty controls | Implemented/tested; conditional replay evaluated |
| Uncertainty-aware cost heuristic | Implemented experimental; no superiority established |
| Strict reservations | Implemented/tested for pinned Llama 3 plain chat protocol; separate replay grid |
| LightGBM + isotonic utilities | Implemented; synthetic plumbing tests only; no real trained/calibrated policy |
| Contextual bandit / RL | Absent |
| Independent local buggy-test evidence execution | Absent in saved local runs; new protocol blocked on tooling/checkouts |
| Large held-out evaluation and statistical noninferiority | Planned, not performed |

Documented Git history:

{{GIT}}

## 14. Limitations and Threats to Validity

Time-25 is a repeatedly inspected development defect. Evidence perturbations, 13 states and 576 policy arms do not increase its independent sample size. The two other saved feasibility bugs provide limited implementation evidence, not a held-out test of the new heuristic. Math-1 fails source-span validation and remains in the feasibility denominator.

The fusion ranker mixes cheap scores with raw Full suspicion using different scales; coarsely quantized suspicion and initial-order ties affect gold rank. Ground truth is used only offline, but candidate coverage and earlier tuning still limit conclusions. Priors are rank-derived, not original semantic probabilities; truncated top-ten lists only reveal censored ranks. Lexical support and reported uncertainty are weak proxies. AST sketches are lossy, and at least one candidate exposes a parse-error feature. Exact span matching does not prove all repository dependencies or runtime behavior were represented.

Temperature zero/seed zero do not guarantee identical behavior on other software/hardware. Replay assumes independent, history-free Full responses and the recorded batching; it cannot predict effects of changed prompts, model weights, or conversational histories. Historical estimated admission is not strict. Strict reservation is conservative and protocol-specific. Verification token savings do not establish latency or monetary savings for the whole localization pipeline.

Qwen and Llama generations differ in model/environment and final ranking implementation. Differences between them cannot be attributed solely to policy changes. Original Qwen raw ledgers are missing. The pre-existing smoke implementation had no complete saved code snapshot; its known input/response artifacts can be audited, but the original uncommitted code state cannot be fully reconstructed from a commit alone.

No statistical confidence intervals, noninferiority guarantees, or paper-novelty claims are justified by these results. No real learned risk calibration is inferred from synthetic tests. Unknown/failed/incomplete runs and historical interrupted usage remain explicitly outside completed-arm comparisons.

## 15. Proposed V3 Research Direction

Keep L1-first rank repair as the experimental architecture and highest-suspicion one-Full as a mandatory strong baseline. Do not adopt the added uncertainty-aware heuristic as a winner. First validate whether cheap evidence carries useful signals across bugs; then evaluate whether a different transparent selection rule or calibrated utility adds value. Distinguish faulty allocation from weak evidence grounding and from mixed-scale ranking effects.

A lightweight future interaction feature could measure the score of the current incumbent, nearby score ties and predicted competitor crossings. This is a hypothesis. Recompute marginal features after each paid action rather than adding singleton utilities. Validate two-action controls before introducing combinatorial optimization or bandit/RL machinery. A learned model is justified only after independent-bug marginal labels show reproducible selection opportunities that simple controls miss.

## 16. Next Experiments and Research Milestones

The pre-existing independent feasibility run is summarized below. Its 64 fresh provider calls consumed 55,341 tokens during its saved execution, with exact within-run cache reuse; those are **not V3 fresh calls**. Time-1 and Lang-1 each contain 25 candidates. Evidence was reused FlexFL trigger text, not locally rerun `failing_tests`. All saved requests/counters/rankings were reaudited. Fixed L1 already matches or beats selective policies at lower cost; no advanced method improves rank.

{{SMOKE}}

Math-1 is excluded from completed ranking comparisons because its `Fraction.negate()` CSV span does not contain that method in the declared buggy checkout. Its manifest entry and error are preserved; no outcome-based replacement was selected.

The new preselected smoke list is **Time-2, Lang-2, Math-2**. Defects4J is not on PATH; their buggy checkouts and corresponding observed `failing_tests` are unavailable. Java 17 alone is insufficient evidence that official project tests can execute. V3 performs a resource preflight and no new inference. [Reproduction instructions](RAVE_FL_Reproducibility.md) provide exact checkout/test/evidence/preflight commands and bounded live commands once prerequisites exist.

Priorities:

1. Prepare official buggy revisions and execute their trigger tests; validate source spans/candidate coverage before inference. Preserve failures and incomplete arms rather than silently choosing easier bugs.
2. Run the bounded three-bug smoke with fixed model digest, prompts/ranker, 14K–20K replay budgets, shared action cache and predeclared physical/call caps. Report reservation-based admission separately from legacy estimates.
3. Freeze the evidence recipe and transparent policy after development. Use the 12 explicitly preselected exploratory bugs (Time/Lang/Math IDs 2–5, including smoke) for feasibility and label collection. They are development data, not held out.
4. Predeclare an additional larger held-out manifest, preferably adding projects; preserve bug-level and project-aware separation from every development case. Fix comparisons and stopping before inspecting held-out outcomes.
5. Only if label diversity and independent bug counts support it, train shallow LightGBM on state-dependent ranking-action outcomes, calibrate on separate bugs, and evaluate once on a held-out split. Report training/calibration counts, utility calibration, paired rankings/tokens, coverage and budget violations. Isotonic fitting is not issue-level safe stopping.

## 17. Reproducibility and Artifact Index

Run the single entry point described in [RAVE_FL_Reproducibility.md](RAVE_FL_Reproducibility.md). It performs zero new model calls, reruns diagnostics/audits/tests, and regenerates English tables, figures and reports from saved artifacts. Existing raw ledgers and manifest hashes are not overwritten. Output directories for experimental datasets must be new.

- [Detailed V2 diagnosis](RAVE_FL_V2_Diagnostic_Analysis.md)
- [V3 conditional replay results](RAVE_FL_V3_Experiment_Results.md)
- [Primary controller comparison](tables/primary_comparison.md)
- [Candidate-level data](tables/candidates.csv), [all policy results](tables/policies.csv), [action transitions](tables/actions.csv), [all decision options](tables/decisions.csv), [pair interactions](tables/pairs.csv)
- [Audits and historical hashes](provenance/artifact_audits.json), [starting state](provenance/starting_state.json), [executed tests](provenance/test_runs.json), [independent resource preflight](provenance/independent_preflight.json)
- [Figure data/source index](tables/figure_manifest.csv), [Git timeline](tables/git_timeline.json), [generated artifact hashes](provenance/generated_sha256.json)
- Source guides: `docs/rave_fl_adaptive_pilot.md`, `docs/rave_fl_rank_repair.md`; original protocols in `configs/`; preserved raw requests/responses under `results/rave_fl_pilot/` and `results/rave_fl_rank_repair/`.

Current test executions:

{{TESTS}}

Three unrelated existing failures remain in `test_pipeline_dependencies.py` (missing `scripts/run_pipeline.sh`, two tests) and `test_agent4sr_format_finalization.py` (seen-entity formatting, one test). Their logs are retained. Repository-wide discovery through downloaded benchmark repositories is not a meaningful project test scope; the scoped project suite is used.

### Figures and detailed captions

{{FIGURES}}
