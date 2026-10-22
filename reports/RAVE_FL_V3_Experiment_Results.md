# RAVE-FL V3 Experiment Results — Conditional Replay

**Execution:** zero new LLM calls; zero fresh inference tokens. This is conditional action replay on one previously inspected bug, Time-25. The 576 comparisons are 6 policies × 3 evidence conditions × 4 budgets × 4 Full-call ceilings × 2 admission modes. They are not 576 independent experiments on different bugs. Saved independent feasibility runs are audited separately below.

## Frozen controls and information boundary

All arms retain the 24-candidate pool, original batch composition of four, condition-specific failure context, `llama3.1:8b` weight digest, temperature 0, seed 0, context 16,384, output cap 160 per candidate, independently prompted Full observations, and `shared_prior_evidence_fusion_v1` with original-order tie-breaking. Gold labels are read after policy actions in a condition. Future Full responses and provider costs are held behind `execute`; planners receive only current observations, input ranks/source features and prompt-derived estimates.

The grid covers 14K/16K/18K/20K, and zero/one/two/four Full ceilings. The zero control is available for every policy. A ceiling is not a forced spend. Full-all is a saved completion reference with a 60K allowance, not a budget-matched 16K controller or accuracy upper bound.

## Policy definitions

Let \(r_i\) be current fused rank, \(r_i^0\) initial rank, \(n\) pool size, \(d_m\) the observed depth of ranker \(m\), and \(s_i,u_i\) raw L1 suspicion and reported uncertainty.

| Policy | Selection rule after complete L1 |
|---|---|
| Fixed L1 | Stop; no Full upgrade |
| Highest L1 suspicion | Choose the maximum raw L1 suspicion; ties by current rank then initial rank. If that candidate cannot be admitted, stop rather than substitute a runner-up |
| Ranker disagreement | Maximize range of three censored rank percentiles; ties by current/initial rank |
| Ranking uncertainty | Maximize the explicit cheap ambiguity proxy below |
| V2 Utility | Unchanged positive top-five discounted-belief improvement per estimated Full token |
| Uncertainty-aware | New transparent development heuristic combining suspicion, disagreement, ambiguity, weak support and remaining-budget-weighted cost |

For an observed rank \(r_{im}\), define \(z_{im}=r_{im}/(d_m+1)\); if missing from the ranker's truncated list, \(z_{im}=1\). Missing means right-censored, not a known actual worst rank. Disagreement is \(D_i=\max_m z_{im}-\min_m z_{im}\). Separate depths are supported and validated; all three depths are ten here. This differs from V2's original disagreement feature, which only compares present ranks or falls back to missing-ranker agreement.

Define adjacent score gaps using current fused scores. Let \(g_*\) be the median strictly positive adjacent gap (or 1 if all tie), \(g_i\) the smallest adjacent gap, and

\[
A_i=\tfrac15\left[\frac{1}{1+g_i/g_*}+\frac{|r_i-r_i^0|}{n-1}
+2\,\operatorname{StdPop}(z_{i1},z_{i2},z_{i3},r_i/(n+1))
+\frac{1}{1+\min(|r_i-1|,|r_i-\min(5,n)|)}+u_i\right].
\]

This is a dimensionless ambiguity proxy, not uncertainty calibrated against failures. Evidence support \(E_i\) is the mean of recorded lexical failure relevance and trace-match features. Both are crude and can miss naming cues.

The new heuristic uses

\[
V_i=\frac{(0.4s_i+0.2D_i+0.25A_i+0.15(1-E_i))(0.5+0.5B_{remain}/B)}{\widehat C_i}.
\]

Weights are transparent development choices, not fitted values or experimentally established optima. The remaining fraction is common to candidates at a given decision, so it rescales the values without independently changing their within-state ordering. Budget still changes the affordable action set. All features are recomputed after each Full upgrade; the candidate's actual previous observation and current scores drive subsequent states. No trained model was introduced.

Controls stop at the call ceiling, when no eligible action fits, or when a Full-confirmed top-1 score strictly exceeds every remaining raw L1 suspicion. That last condition is a cheap heuristic, not a guarantee. Highest Suspicion has no first-action suspicion cutoff. V2 Utility retains its nonpositive-proxy stopping rule. Differences in stopping are explicitly part of the defined policies; maximum-one controls isolate the first selection without forced multi-call spending.

## Primary comparison: matched maximum one Full

Historical estimate admission reproduces V2's operational comparison:

| condition | policy | tokens | rank | MRR | full_calls |
|---|---|---|---|---|---|
| Original | fixed_l1 | 11298 | 4 | 0.25 | 0 |
| Original | highest_suspicion | 12516 | 1 | 1.0 | 1 |
| Original | ranker_disagreement | 12031 | 3 | 0.3333333333333333 | 1 |
| Original | ranking_ambiguity | 11963 | 4 | 0.25 | 1 |
| Original | v2_utility | 12516 | 1 | 1.0 | 1 |
| Original | uncertainty_aware | 11963 | 4 | 0.25 | 1 |
| Masked | fixed_l1 | 11710 | 23 | 0.043478260869565216 | 0 |
| Masked | highest_suspicion | 12499 | 23 | 0.043478260869565216 | 1 |
| Masked | ranker_disagreement | 12499 | 23 | 0.043478260869565216 | 1 |
| Masked | ranking_ambiguity | 12473 | 22 | 0.045454545454545456 | 1 |
| Masked | v2_utility | 11710 | 23 | 0.043478260869565216 | 0 |
| Masked | uncertainty_aware | 12406 | 22 | 0.045454545454545456 | 1 |
| Masked + Numeric | fixed_l1 | 12292 | 23 | 0.043478260869565216 | 0 |
| Masked + Numeric | highest_suspicion | 13185 | 23 | 0.043478260869565216 | 1 |
| Masked + Numeric | ranker_disagreement | 13185 | 23 | 0.043478260869565216 | 1 |
| Masked + Numeric | ranking_ambiguity | 13158 | 22 | 0.045454545454545456 | 1 |
| Masked + Numeric | v2_utility | 12292 | 23 | 0.043478260869565216 | 0 |
| Masked + Numeric | uncertainty_aware | 13185 | 23 | 0.043478260869565216 | 1 |

Original Highest Suspicion and Utility both reach rank 1 at 12,516. Disagreement reaches rank 3 for fewer tokens but does not beat the best original result. Ranking uncertainty and the new heuristic reach rank 4 while spending more than Fixed L1. Under masking, the best single-action rank changes shown are 23→22, far from Top-5. Numeric augmentation does not rescue Utility or gold L1 scores. There is no demonstrated advanced-policy advantage worth claiming over the strongest relevant simple baseline.

The new heuristic's masked rank-22 result at 12,406 uses 93 fewer accounted tokens than Highest Suspicion's rank-23 result at 12,499, and 67 fewer than Ranking Uncertainty's rank-22 result. This is a measurable local advantage over those one-call controls, despite the weak final rank. It still costs more than Fixed L1/Utility at rank 23 and does not recover the gold opportunity or establish useful generalization. Do not elevate that small movement into an accuracy-preservation finding.

## Strict budget reservations

The following 16K cap-one results reserve conservative prompt/token upper bounds without inspecting future actual cost:

| condition | policy | tokens | rank | MRR | full_calls | complete_l1 |
|---|---|---|---|---|---|---|
| Original | fixed_l1 | 11298 | 4 | 0.25 | 0 | True |
| Original | highest_suspicion | 11298 | 4 | 0.25 | 0 | True |
| Original | ranker_disagreement | 12031 | 3 | 0.3333333333333333 | 1 | True |
| Original | ranking_ambiguity | 11963 | 4 | 0.25 | 1 | True |
| Original | v2_utility | 11298 | 4 | 0.25 | 0 | True |
| Original | uncertainty_aware | 11963 | 4 | 0.25 | 1 | True |
| Masked | fixed_l1 | 11710 | 23 | 0.043478260869565216 | 0 | True |
| Masked | highest_suspicion | 12499 | 23 | 0.043478260869565216 | 1 | True |
| Masked | ranker_disagreement | 12499 | 23 | 0.043478260869565216 | 1 | True |
| Masked | ranking_ambiguity | 12473 | 22 | 0.045454545454545456 | 1 | True |
| Masked | v2_utility | 11710 | 23 | 0.043478260869565216 | 0 | True |
| Masked | uncertainty_aware | 12406 | 22 | 0.045454545454545456 | 1 | True |
| Masked + Numeric | fixed_l1 | 10567 | 22 | 0.045454545454545456 | 0 | False |
| Masked + Numeric | highest_suspicion | 10567 | 22 | 0.045454545454545456 | 0 | False |
| Masked + Numeric | ranker_disagreement | 10567 | 22 | 0.045454545454545456 | 0 | False |
| Masked + Numeric | ranking_ambiguity | 10567 | 22 | 0.045454545454545456 | 0 | False |
| Masked + Numeric | v2_utility | 10567 | 22 | 0.045454545454545456 | 0 | False |
| Masked + Numeric | uncertainty_aware | 10567 | 22 | 0.045454545454545456 | 0 | False |

Strict mode uses the pinned template/byte-BPE reservation in `budget_reservation.py` and checks every result against input/output bounds. All recorded strict replay arms respect their budgets; all costs are charged after selection. Historical estimate arms are labeled estimate-only even when no actual overrun occurs. Original Highest Suspicion cannot admit its maximum candidate with the strict 16K balance, so it makes no upgrade. It is not silently redefined as "highest affordable suspicion". At 20K, the strict one-Full Highest Suspicion and Utility recover the original 12,516/rank-1 result. Incomplete strict L1 coverage in the numeric 16K condition is explicitly flagged and excluded from claims about policies that start after global L1.

Conservative admission can cost ranking opportunity even when actual realized costs would have fit. Lowering that conservatism requires an independently valid prompt tokenizer/counting method, not retrospective use of cached response totals. A future service-enforced billing cap or exact tokenizer can be compared as a separate implementation change.

## Multi-upgrade and equal-budget controls

The full grid is [tables/policies.csv](tables/policies.csv), with every action transition in [tables/actions.csv](tables/actions.csv) and every Full candidate option in [tables/decisions.csv](tables/decisions.csv). Compare rows at identical evidence, admission, budget and maximum Full count. No-upgrade controls establish whether additional spending improves anything. Two- and four-upgrade ceilings expose nonmonotonic rank effects and stop-rule behavior. V2 Utility may stop after one action even when four are permitted; that alone is not a fair advantage over a baseline artificially forced to four.

At original 20K/max-four historical admission, Highest Suspicion ends rank 1 at 15,191 tokens; Utility stops at the same rank after one action and 12,516. The one-call comparison already shows that Highest Suspicion can achieve Utility's cost/result without the extra cap. The new heuristic at max-four spends 14,081 and ends rank 2. Under masking, no tested policy reaches Top-5 through four calls; the rank improvements arise from demoting competitors, not buying the gold Full action. No missing response is fabricated: the complete independent Full cache covers these exact actions. Changed source/prompts or uncached actions are refused as unavailable.

## Interactions and oracle ceilings

282 conditional action rows across 13 partial states: 22 positive, 13 negative, 247 zero marginal MRR changes. All rows belong to one defect.

Pair utility is state-dependent and nonadditive. Details, the strongest original pair, and the absence of the stronger "two individually positive actions become net harmful" pattern are reported in [V2 diagnostics](RAVE_FL_V2_Diagnostic_Analysis.md). Oracle subsets are gold/future-informed diagnostic ceilings, not online results and not evidence that the new features can find the best action.

## Saved independent feasibility, not fresh V3 inference

| bug | policy | tokens | rank | MRR | full_calls |
|---|---|---|---|---|---|
| Time-1 | full | 16562 | 1 | 1.0 | 25 |
| Time-1 | fixed_l1 | 11218 | 1 | 1.0 | 0 |
| Time-1 | highest_suspicion | 12340 | 1 | 1.0 | 1 |
| Time-1 | ranker_disagreement | 11642 | 1 | 1.0 | 1 |
| Time-1 | ranking_ambiguity | 11664 | 1 | 1.0 | 1 |
| Time-1 | v2_utility | 11664 | 1 | 1.0 | 1 |
| Lang-1 | full | 18064 | 2 | 0.5 | 25 |
| Lang-1 | fixed_l1 | 9497 | 4 | 0.25 | 0 |
| Lang-1 | highest_suspicion | 10010 | 4 | 0.25 | 1 |
| Lang-1 | ranker_disagreement | 10067 | 4 | 0.25 | 1 |
| Lang-1 | ranking_ambiguity | 10776 | 4 | 0.25 | 1 |
| Lang-1 | v2_utility | 9497 | 4 | 0.25 | 0 |

These existing Time-1/Lang-1 observations were present uncommitted at task start and are reaudited, not rerun. The saved manifest records Time-1, Lang-1 and Math-1 with official buggy commit identifiers and a preselection rule. Its pre-outcome timing cannot be independently established from Git because these files were uncommitted. Exact source/evidence/request/response/counter/ranking checks pass for the two completed cases. The saved implementation's entire code state was not snapshotted; its earlier chronology is undocumented beyond run metadata and current starting file hashes.

The manifest selects the smallest active ID per project excluding Time-25. Math-1 fails method-span validation and remains an honest unsuccessful feasibility case. All saved evidence is FlexFL trigger text rather than locally rerun failing tests. Thus these results assess bounded input/inference feasibility under reused evidence, not the complete fresh evidence protocol requested for the next phase. Fixed L1 already matches every selective outcome at lower tokens; no selective improvement is demonstrated. The new V3 heuristic was not freshly evaluated on these bugs.

The preselected **previously unused** list for the next smoke is Time-2, Lang-2, Math-2. The current environment lacks Defects4J on PATH and those prepared buggy checkouts/failure files. No fresh sweep is launched. The 12-bug exploratory plan (IDs 2–5 across Time/Lang/Math) must remain development data, with a separate later held-out set.

## Implementation and validation impact

Added: reservation backend; all-candidate/action/decision/pair datasets; six-policy diagnostics; historic/smoke audits; resource preflight; professional report/figure regeneration and single pipeline entry point. Reviewed and extended: pre-existing selection/smoke modules. Fixed: replay now verifies the actual execution prompt, not just an ID tuple. V2 Utility, all shared ranking coefficients/tie-breaking, sketches and L1/Full prompts remain unchanged.

| label | command | summary | returncode |
|---|---|---|---|
| RAVE-FL focused suite | /Users/mohammadmahdi/.pyenv/versions/3.11.0/bin/python3 -m pytest -q tests/test_rave_fl.py tests/test_rave_fl_adapter.py tests/test_rave_fl_evidence.py tests/test_rave_fl_expanded.py tests/test_rave_fl_iteration_d.py tests/test_rave_fl_phase3.py tests/test_rave_fl_pilot.py tests/test_rave_fl_rank_repair.py tests/test_rave_fl_v3.py | 91 passed in 2.44s | 0 |
| Scoped project suite | /Users/mohammadmahdi/.pyenv/versions/3.11.0/bin/python3 -m pytest -q tests --ignore=tests/test_pipeline_dependencies.py --ignore=tests/test_agent4sr_format_finalization.py | 111 passed in 1.81s | 0 |
| Known unrelated failures | /Users/mohammadmahdi/.pyenv/versions/3.11.0/bin/python3 -m pytest -q tests/test_pipeline_dependencies.py tests/test_agent4sr_format_finalization.py | 3 failed, 1 passed in 0.19s | 1 |

Malformed-response handling and all paid usage remain covered by existing tests. Focused V3 tests cover conservative refusal before revealing a cached answer, model/template identity, byte bounds, visible counter violations, no substitution for an unaffordable maximum, all-option audit completeness, deterministic zero-upgrade controls, leakage boundaries and unverified candidate retention. The three existing unrelated project failures remain documented and unchanged.

## Parallel branch boundary

A final safety fetch found 32 additional collaborator commits on the old remote branch (head `e6e9fd9`), including grounded repair, live-localizer and offline bandit/SWE-bench extensions. They are preserved unmerged and are not part of this six-policy experiment. See the progress report and `provenance/parallel_remote_branch.json`; do not interpret the evaluated branch's absent capabilities as an assertion about those parallel modules.

## Research decision

Continue with the L1-first architecture and the simplest strong controls. The new heuristic is a negative development result. Prioritize reproducible buggy-test evidence and candidate-span compatibility on independent bugs before training utility or adding a complex optimizer. Preserve oracle labels for diagnosis, keep them out of features, and require a measurable advantage over Highest Suspicion and Fixed L1 at fair actual cost before advancing a learned-policy claim.

