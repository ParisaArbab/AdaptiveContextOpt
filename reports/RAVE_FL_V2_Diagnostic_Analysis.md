# RAVE-FL V2 Diagnostic Analysis

**Scope:** Time-25, one development defect, 24 fixed candidates, local `llama3.1:8b`, temperature 0, seed 0, 16,384 context, up to 160 output tokens per candidate. Scores below are ranking/suspicion proxies, not calibrated probabilities. Every cost is verification-stage provider input plus output tokens. Full outcomes, gold identity, and realized marginal MRR appear only in offline analysis.

## Evidence and source hierarchy

The source runs are `results/rave_fl_pilot/time25_llama31_8b/run.json` (original independent Full/L1 references), `results/rave_fl_rank_repair/live_original/`, `live_masked/`, and `live_normalized_masked/`. Raw provider ledgers and exact prompt checks support the Llama measurements. Historical Qwen summaries belong to a different experimental generation; they are not used to explain a Llama policy effect. The new datasets are conditional replay, not newly generated model answers.

## 1. Reconstructing the original repair

Candidate C22 is `org.joda.time.DateTimeZone.getOffsetFromLocal(long)`. It starts at rank 22, with rank-derived prior approximately 0.2. All candidates remain in the ranking, including those without Full verification. After global L1, its raw suspicion 0.9, reported uncertainty 0.1, and prior anchor 0.2 give

\[
S_{22}=(1-0.65(1-0.1))0.2+0.65(1-0.1)0.9=0.6095.
\]

That places it fourth behind `DateTime(long,DateTimeZone)` (0.704), `ZoneInfoCompiler.compile(File,File[])` (0.68), and `CachedDateTimeZone.nextTransition(long)` (0.632). The next method, `AbstractInstant.toString()`, scores 0.608: a small 0.0015 gap below C22. The complete pool and ranks follow in Section 6.

The existing Utility chooses C22 itself, not a competing candidate. Its pre-action projection substitutes raw L1 suspicion 0.9 for fused 0.6095, predicts rank 1, and assigns positive discounted-belief gain. The provider Full result is 0.8, costing 1,195 input plus 23 output tokens, 1,218 total. The frozen ranker directly uses 0.8 and moves C22 to rank 1. Every other candidate's score is unchanged by this action. The realized MRR change is 1 − 1/4 = 0.75, and the policy total is 11,298 + 1,218 = 12,516.

These exact action/cost details are checked against `tables/actions.csv` and provider ledgers. If a regenerated source ledger changes, the generated tables and audit must be consulted rather than treating the explanatory prose as a substitute for data.

### Full-all versus mixed L1-plus-Full

The competitor `DateTimeZone.convertLocalToUTC(long,boolean)` has L1 suspicion 0.4, uncertainty 0.6, cheap score 0.437, and Full suspicion 0.8. In the mixed state it stays below C22. In Full-all it ties C22 at 0.8 and precedes it under the frozen original-order tie-breaker, giving C22 rank 2. Thus the mixed state's better gold rank follows exactly from the score-replacement rule and which competitors remain cheap. Full-all is not an oracle upper bound.

There is no evidence here that Full increased C22's raw suspicion: it decreased from 0.9 to 0.8. It increased C22's *fusion score* because Full bypasses cheap prior shrinkage. This scale difference is part of the observed ranking mechanism. The ledger does not reveal an internal causal reasoning process or prove semantic correctness of the explanation.

### Decision-time and retrospective values

| condition | initial_rank | input_ranks | current_rank | l1_suspicion | l1_uncertainty | current_score | gap_above | gap_below | disagreement | evidence_support | full_rank_offline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | 22 | SBIR 9, Ochiai absent, BoostN 9 | 4 | 0.9 | 0.1 | 0.6095 | 0.022499999999999964 | 0.0014999999999999458 | 0.18181818181818177 | 0.05188679245283019 | 2 |
| Masked | 22 | SBIR 9, Ochiai absent, BoostN 9 | 23 | 0.4 | 0.6 | 0.252 | 0.03500000000000003 | 0.0955 | 0.18181818181818177 | 0.0330188679245283 | 6 |
| Masked + Numeric | 22 | SBIR 9, Ochiai absent, BoostN 9 | 23 | 0.4 | 0.6 | 0.252 | 0.03500000000000003 | 0.0955 | 0.18181818181818177 | 0.04245283018867924 | 9 |

| condition | gain_proxy | utility_per_token | projected_rank | predicted_tokens | actual_tokens_offline | strict_reservation |
|---|---|---|---|---|---|---|
| Original | 0.05667 | 0.00002 | 1 | 2520 | 1218 | 4942 |
| Masked | 0 | 0 | 18 | 2516 | 1289 | 4934 |
| Masked + Numeric | 0 | 0 | 19 | 2716 | 1386 | 5336 |

Raw input ranker diagnostics (deeper ranks are not used by the evaluated top-ten controller features):

| condition | SBIR_full_list_rank | SBIR_recorded_score | Ochiai_full_list_rank | Ochiai_recorded_score | BoostN_full_list_rank | BoostN_recorded_score |
|---|---|---|---|---|---|---|
| Original | 9 | Unavailable | 53 | Unavailable | 9 | 0.71168655 |
| Masked | 9 | Unavailable | 53 | Unavailable | 9 | 0.71168655 |
| Masked + Numeric | 9 | Unavailable | 53 | Unavailable | 9 | 0.71168655 |

`actual_tokens_offline` is never used for selection. `strict_reservation` is computed from prompt bytes and output/context bounds before Full, without reading that response. The historical predicted costs use the unchanged character-based estimate. The new censored disagreement metric and V2's original feature are distinct: C22 has V2 `ranker_disagreement = 0` because its two observed ranks are both 9, while the new three-signal censored range is 2/11 because Ochiai's top-ten entry is absent.

## 2. Why masked evidence fails

Masking removes candidate-name substrings from test/frame method identifiers, preserving assertion text. It also masks other candidate names (`DateTime` and generic `test`) wherever the procedure matches, so the perturbation cannot be attributed solely to one gold-name replacement. Class/file/line and scenario suffixes remain. The model receives opaque aliases, not their mapping.

C22's L1 suspicion drops to 0.4, uncertainty rises to 0.6, and score becomes

\[
S_{22}=(1-0.65(1-0.6))0.2+0.65(1-0.6)0.4=0.252.
\]

Its L1 rank falls to 23. The top L1 distractors retain large scores, including the DateTime constructor, compiler, cached transition, and instant-formatting candidates. The first-versus-second score margin is essentially unchanged; a confident-looking head therefore does not measure the unseen gold tail's risk. C22's local gaps become 0.035 above and 0.0955 below. It is neither near the top-five boundary nor strongly selected by cross-ranker disagreement.

Its ranker inputs do not change under masking: SBIR 9, BoostN 9, Ochiai absent. V2's ranker disagreement remains zero and agreement remains 2/3. The full source CSVs also contain deeper positions and BoostN raw suspiciousness; these are exported separately in `tables/candidates.csv`. SBIR/Ochiai CSVs have no numeric score column. The evaluated policies deliberately use the existing top-ten signals; a method absent there is censored for that feature even when its deeper CSV position can be reported diagnostically. Source complexity, sketch size, parse/truncation state and source-token proxies also remain unchanged. C22 is parse-valid, not truncated, complexity 4, four calls, L1 approximately 173 planning tokens versus source 662. Lexical `failure_relevance` drops from about 0.10377 to 0.06604; the existing regex `trace_match` is zero in both contexts because the method name occurs within longer test identifiers rather than as the exact standalone leaf. This feature therefore does not faithfully measure all naming cues available to the LLM.

The cheap raw-suspicion projection moves C22 only from rank 23 to 18 (19 under numeric augmentation), outside Utility's top-five objective. It yields exactly zero gain, even though the cached Full suspicion is 0.8. All affordable Utility options have nonpositive gains, so the explicit stop rule buys no Full inspections. Reported uncertainty 0.6 is not treated as a calibrated chance of a large positive Full surprise. This is a concrete blind spot: a point prediction based on weak L1 evidence cannot value beneficial out-of-support movement into the top five.

The uncertainty/support proxies cannot independently establish *why* the model weakened the observation. They identify weak observations and an inadequate value/stopping proxy. Selection bias is visible in the projection's emphasis on candidates already near the head; independent outcomes for all 24 methods make it possible to diagnose that bias offline. It is a hypothesis, rather than a demonstrated internal mechanism, that the model relied heavily on direct naming cues.

### Why Full-all falls from rank 2 to 6, then 9

| condition | full_scores_above_gold | equal_score_candidates_ahead_by_prior | gold_full_score | l1_top2_margin |
|---|---|---|---|---|
| Original | 0 | 1 | 0.8 | 0.024 |
| Masked | 0 | 5 | 0.8 | 0.024 |
| Masked + Numeric | 0 | 8 | 0.8 | 0.024 |

C22's Full suspicion remains 0.8. Its worse Full-all rank is exactly explained by changed competitors under the shared sort rule: the earlier constructor/test/conversion/base-date candidates receive equal or higher scores. The candidate table gives every affected method. This is not simply an allocation failure, because Full-all inspects every method yet still loses Top-5 under masked evidence.

### Oracle opportunity versus an online-selectable action

Masked one-Full oracle replay upgrades C22 and reaches rank 1 at 12,999 tokens. It uses the future Full result and gold labels to pick the action; this is an offline diagnostic ceiling. Highest raw L1 suspicion chooses a different, high-head-suspicion candidate and remains at rank 23. Disagreement also favors a different candidate; uncertainty-aware selection reaches only rank 22 with one call. All tested online policies miss the masked Top-5 opportunity through four upgrades. Exact budget/call-cap tables are in [the V3 report](RAVE_FL_V3_Experiment_Results.md).

An online policy could see C22's source sketch, low prior rank, 0.4 suspicion, 0.6 reported uncertainty, weak lexical support and ranker input positions. It could not know its unseen Full score 0.8 or gold identity. Oracle opportunity alone does not establish that a realizable low-cost policy can select that action reliably. Higher budget by itself does not repair Utility's zero-gain stop rule.

## 3. Numeric augmentation: explicitly post-hoc

After the masked failure, a deterministic parser **prepended** three assertion summaries to the complete masked evidence. Each states expected +04:00, observed +03:00, difference +3,600,000 milliseconds. It retains the raw ellipsized assertion text, does not reconstruct missing dates, and adds no candidate identity or diagnosis. This increases the context from 1,502 to 1,904 characters. The exact prompts remain in the two conditions' `_calls.jsonl` files and input snapshots.

The following counts compare all 24 candidate scores between masked and numeric conditions:

| l1_suspicion_changed | l1_uncertainty_changed | l1_fused_score_changed | full_suspicion_changed |
|---|---|---|---|
| 1 | 1 | 1 | 3 |

The faulty method's L1 suspicion remains 0.4, uncertainty 0.6, fused score 0.252, and rank 23. Its Full suspicion remains 0.8. Numerical clarification changes some competing Full scores and wording but does not produce a stronger cheap signal for the gold method. Full-all worsens from rank 6 to 9; L1 and Utility remain rank 23; cost increases in every arm. The numeric gold Full costs 1,386 rather than 1,289 tokens, and the oracle mixed state costs 13,678. This is a failed post-hoc development intervention, not independent validation or evidence that numeric detail is generally harmful.

Short raw explanations illustrate the difference between an observed score and a plausible narrative:

| condition | l1_explanation | full_explanation_offline |
|---|---|---|
| Original | Possible issue with time zone offset calculation | method handles DST transitions and returns offset after transition |
| Masked | Possible issue with offset adjustment in getOffsetFromLocal | method handles DST transitions and returns offset for instant after transition |
| Masked + Numeric | Incorrect offset adjustment | method returns incorrect UTC offset |

A more specific explanation in the numeric Full response does not improve its score or final position. Neither the ledger nor this controlled comparison proves that the model actually performs arithmetic correctly or grounds the offset change in the faulty branch. Prompt length, repeated numbers and altered evidence presentation also change together; do not infer a unique causal formatting mechanism.

## 4. Action interactions

282 conditional action rows across 13 partial states: 22 positive, 13 negative, 247 zero marginal MRR changes. All rows belong to one defect.

Positive actions can promote the gold method or demote competitors; negative actions can promote a competitor above it. Each marginal label is conditional on a recorded state, not a property of the method alone. Full inspection can therefore harm ranking. The strongest pair interaction in the original all-L1 starting state is:

| first | second | first_singleton_delta | second_singleton_delta | pair_delta_MRR | interaction_MRR | pair_gold_rank |
|---|---|---|---|---|---|---|
| org.joda.time.DateTimeZone.convertLocalToUTC(long,boolean) | org.joda.time.DateTimeZone.getOffsetFromLocal(long) | -0.04999999999999999 | 0.75 | 0.25 | -0.45 | 2 |

Let \(G(A)=MRR(A)-MRR(\varnothing)\). The interaction is

\[
I(i,j)=G(\{i,j\})-G(\{i\})-G(\{j\}).
\]

For the gold/conversion pair, the positive gold upgrade and negative competitor upgrade combine nonadditively. The original action-row dataset also changes C22's marginal MRR gain from 0.75 in all-L1 to 0 in a state where earlier distractor demotions already put it first. `convertLocalToUTC` ranges from −0.05 to −0.5 across recorded prefixes. Such changes make static independent-action utility unreliable.

The exhaustive original pair table contains 0 examples in which **both** singleton gains are strictly positive but their combination is net negative against all-L1. That stronger scenario is possible in richer ranking systems but is not demonstrated by this dataset. Do not conflate negative interaction with that claim. A future interaction-aware extension should begin with incumbent/tie/score-crossing features and sequential feature recomputation, not an unvalidated complex optimizer.

## 5. Measured findings, hypotheses, and unresolved questions

| Kind | Finding |
|---|---|
| Measured | Original Utility upgrades the faulty method; raw highest suspicion selects the same method/cost/result |
| Measured | Score shrinkage removal and competitor tie-breaking explain mixed rank 1 versus Full-all rank 2 |
| Measured | Masked L1 gold suspicion/uncertainty deteriorate; Utility's top-five projection is zero-gain and stops |
| Measured | Full gold score stays 0.8; competitors account for Full-all rank regressions |
| Oracle only | Gold-informed one-Full mixed states can reach rank 1 in masked/numeric conditions |
| Hypothesis | Better evidence-grounding signals could improve low-cost selection of beneficial surprises |
| Unproven | A trained or heuristic policy can recover that opportunity across independent bugs |
| Unproven | Original evidence validity, calibrated uncertainty, overall accuracy preservation, or statistical noninferiority |

## 6. All-candidate reconstruction

The IDs C01–C24 are the unchanged initial pool order. The table includes the full supplied method signature, initial rank, all-L1 rank/scores, Full-all rank/scores, pre-action predicted Full cost and retrospective provider cost. Full-all columns are offline outcomes, not selectable features. Exact per-ranker ranks, support features, pre/post action states, and all decision options are available in the CSVs for detailed reuse.

### Original

| short_id | initial_rank | current_rank | l1_suspicion | l1_uncertainty | current_score | full_rank_offline | full_suspicion_offline | predicted_full_tokens | full_tokens_offline |
|---|---|---|---|---|---|---|---|---|---|
| C01 DateTime.DateTime(long,DateTimeZone) | 1 | 1 | 0.8 | 0.2 | 0.704 | 3 | 0.0 | 1428 | 733 |
| C02 ZoneInfoCompiler.test(String,DateTimeZone) | 2 | 8 | 0.1 | 0.9 | 0.5675 | 4 | 0.0 | 2548 | 1115 |
| C03 ZoneInfoCompiler.printUsage() | 3 | 9 | 0.0 | 1.0 | 0.55 | 5 | 0.0 | 1430 | 694 |
| C04 DateTime.DateTime(int,int,int,int,int,DateTimeZone) | 4 | 7 | 0.6 | 0.4 | 0.5695 | 6 | 0.0 | 1614 | 806 |
| C05 ZoneInfoCompiler.compile(File,File[]) | 5 | 2 | 0.8 | 0.2 | 0.68 | 7 | 0.0 | 3080 | 1322 |
| C06 BaseDateTime.BaseDateTime(long,DateTimeZone) | 6 | 13 | 0.4 | 0.6 | 0.474 | 8 | 0.0 | 1450 | 742 |
| C07 BasicGJChronology.setYear(long,int) | 7 | 10 | 0.6 | 0.4 | 0.5389999999999999 | 9 | 0.0 | 1638 | 804 |
| C08 PrecalculatedZone.create(String,boolean,ArrayList<Transition>,DSTZone) | 8 | 14 | 0.2 | 0.8 | 0.46099999999999997 | 10 | 0.0 | 3620 | 1518 |
| C09 CachedDateTimeZone.nextTransition(long) | 9 | 3 | 0.8 | 0.2 | 0.632 | 11 | 0.0 | 1248 | 620 |
| C10 DateTimeZone.convertLocalToUTC(long,boolean) | 10 | 16 | 0.4 | 0.6 | 0.43699999999999994 | 1 | 0.8 | 2622 | 1150 |
| C11 DateTimeZoneBuilder.readFrom(DataInput,String) | 11 | 18 | 0.2 | 0.8 | 0.4175 | 12 | 0.0 | 1646 | 802 |
| C12 DateTimeZone.parseOffset(String) | 12 | 12 | 0.6 | 0.4 | 0.478 | 13 | 0.0 | 1542 | 732 |
| C13 AbstractInstant.toString() | 13 | 5 | 0.8 | 0.2 | 0.6080000000000001 | 14 | 0.0 | 1318 | 665 |
| C14 DateTimeZoneBuilder.toDateTimeZone(String,boolean) | 14 | 20 | 0.4 | 0.6 | 0.4 | 15 | 0.0 | 2626 | 1184 |
| C15 DateTimeZone.getDefaultProvider() | 15 | 21 | 0.2 | 0.8 | 0.3305 | 16 | 0.0 | 1902 | 881 |
| C16 BaseDateTime.BaseDateTime(int,int,int,int,int,int,int,DateTimeZone) | 16 | 15 | 0.6 | 0.4 | 0.4475 | 17 | 0.0 | 1732 | 850 |
| C17 DateTimeFormatter.parseLocalDateTime(String) | 17 | 6 | 0.8 | 0.2 | 0.5840000000000001 | 18 | 0.0 | 2072 | 967 |
| C18 DateTimeZone.forID(String) | 18 | 22 | 0.4 | 0.6 | 0.32600000000000007 | 19 | 0.0 | 1910 | 923 |
| C19 ISODateTimeFormat.dateTime() | 19 | 23 | 0.2 | 0.8 | 0.28700000000000003 | 20 | 0.0 | 1468 | 731 |
| C20 CachedDateTimeZone.createInfo(long) | 20 | 19 | 0.6 | 0.4 | 0.41700000000000004 | 21 | 0.0 | 1468 | 729 |
| C21 ISODateTimeFormat.tTime() | 21 | 11 | 0.8 | 0.2 | 0.536 | 22 | 0.0 | 1512 | 765 |
| C22 DateTimeZone.getOffsetFromLocal(long) | 22 | 4 | 0.9 | 0.1 | 0.6095 | 2 | 0.8 | 2520 | 1218 |
| C23 ISODateTimeFormat.time() | 23 | 17 | 0.7 | 0.3 | 0.42749999999999994 | 23 | 0.0 | 1508 | 754 |
| C24 Fraction.printTo(StringBuffer,long,Chronology,int,DateTimeZone,Locale) | 24 | 24 | 0.4 | 0.6 | 0.21500000000000002 | 24 | 0.0 | 1400 | 673 |

### Masked

| short_id | initial_rank | current_rank | l1_suspicion | l1_uncertainty | current_score | full_rank_offline | full_suspicion_offline | predicted_full_tokens | full_tokens_offline |
|---|---|---|---|---|---|---|---|---|---|
| C01 DateTime.DateTime(long,DateTimeZone) | 1 | 1 | 0.8 | 0.2 | 0.704 | 1 | 0.8 | 1424 | 789 |
| C02 ZoneInfoCompiler.test(String,DateTimeZone) | 2 | 8 | 0.4 | 0.6 | 0.548 | 2 | 0.8 | 2544 | 1207 |
| C03 ZoneInfoCompiler.printUsage() | 3 | 7 | 0.0 | 1.0 | 0.55 | 7 | 0.0 | 1426 | 763 |
| C04 DateTime.DateTime(int,int,int,int,int,DateTimeZone) | 4 | 6 | 0.6 | 0.4 | 0.5695 | 3 | 0.8 | 1610 | 872 |
| C05 ZoneInfoCompiler.compile(File,File[]) | 5 | 2 | 0.8 | 0.2 | 0.68 | 8 | 0.0 | 3076 | 1391 |
| C06 BaseDateTime.BaseDateTime(long,DateTimeZone) | 6 | 11 | 0.4 | 0.6 | 0.474 | 9 | 0.0 | 1446 | 810 |
| C07 BasicGJChronology.setYear(long,int) | 7 | 9 | 0.6 | 0.4 | 0.5389999999999999 | 10 | 0.0 | 1634 | 873 |
| C08 PrecalculatedZone.create(String,boolean,ArrayList<Transition>,DSTZone) | 8 | 12 | 0.2 | 0.8 | 0.46099999999999997 | 11 | 0.0 | 3616 | 1585 |
| C09 CachedDateTimeZone.nextTransition(long) | 9 | 3 | 0.8 | 0.2 | 0.632 | 12 | 0.0 | 1244 | 696 |
| C10 DateTimeZone.convertLocalToUTC(long,boolean) | 10 | 14 | 0.4 | 0.6 | 0.43699999999999994 | 4 | 0.8 | 2618 | 1219 |
| C11 DateTimeZoneBuilder.readFrom(DataInput,String) | 11 | 15 | 0.2 | 0.8 | 0.4175 | 13 | 0.0 | 1642 | 870 |
| C12 DateTimeZone.parseOffset(String) | 12 | 18 | 0.1 | 0.9 | 0.38050000000000006 | 14 | 0.0 | 1538 | 800 |
| C13 AbstractInstant.toString() | 13 | 4 | 0.8 | 0.2 | 0.6080000000000001 | 15 | 0.0 | 1314 | 734 |
| C14 DateTimeZoneBuilder.toDateTimeZone(String,boolean) | 14 | 17 | 0.4 | 0.6 | 0.4 | 16 | 0.0 | 2622 | 1252 |
| C15 DateTimeZone.getDefaultProvider() | 15 | 20 | 0.2 | 0.8 | 0.3305 | 17 | 0.0 | 1898 | 949 |
| C16 BaseDateTime.BaseDateTime(int,int,int,int,int,int,int,DateTimeZone) | 16 | 13 | 0.6 | 0.4 | 0.4475 | 5 | 0.8 | 1728 | 930 |
| C17 DateTimeFormatter.parseLocalDateTime(String) | 17 | 5 | 0.8 | 0.2 | 0.5840000000000001 | 18 | 0.0 | 2068 | 1036 |
| C18 DateTimeZone.forID(String) | 18 | 21 | 0.4 | 0.6 | 0.32600000000000007 | 19 | 0.0 | 1906 | 992 |
| C19 ISODateTimeFormat.dateTime() | 19 | 22 | 0.2 | 0.8 | 0.28700000000000003 | 20 | 0.0 | 1464 | 799 |
| C20 CachedDateTimeZone.createInfo(long) | 20 | 16 | 0.6 | 0.4 | 0.41700000000000004 | 21 | 0.0 | 1464 | 797 |
| C21 ISODateTimeFormat.tTime() | 21 | 10 | 0.8 | 0.2 | 0.536 | 22 | 0.0 | 1508 | 833 |
| C22 DateTimeZone.getOffsetFromLocal(long) | 22 | 23 | 0.4 | 0.6 | 0.252 | 6 | 0.8 | 2516 | 1289 |
| C23 ISODateTimeFormat.time() | 23 | 19 | 0.6 | 0.4 | 0.356 | 23 | 0.0 | 1504 | 823 |
| C24 Fraction.printTo(StringBuffer,long,Chronology,int,DateTimeZone,Locale) | 24 | 24 | 0.2 | 0.8 | 0.1565 | 24 | 0.0 | 1396 | 750 |

### Masked + Numeric

| short_id | initial_rank | current_rank | l1_suspicion | l1_uncertainty | current_score | full_rank_offline | full_suspicion_offline | predicted_full_tokens | full_tokens_offline |
|---|---|---|---|---|---|---|---|---|---|
| C01 DateTime.DateTime(long,DateTimeZone) | 1 | 1 | 0.8 | 0.2 | 0.704 | 1 | 0.8 | 1626 | 893 |
| C02 ZoneInfoCompiler.test(String,DateTimeZone) | 2 | 8 | 0.4 | 0.6 | 0.548 | 2 | 0.8 | 2746 | 1300 |
| C03 ZoneInfoCompiler.printUsage() | 3 | 7 | 0.0 | 1.0 | 0.55 | 10 | 0.0 | 1628 | 866 |
| C04 DateTime.DateTime(int,int,int,int,int,DateTimeZone) | 4 | 6 | 0.6 | 0.4 | 0.5695 | 3 | 0.8 | 1812 | 973 |
| C05 ZoneInfoCompiler.compile(File,File[]) | 5 | 2 | 0.8 | 0.2 | 0.68 | 11 | 0.0 | 3278 | 1494 |
| C06 BaseDateTime.BaseDateTime(long,DateTimeZone) | 6 | 12 | 0.4 | 0.6 | 0.474 | 4 | 0.8 | 1646 | 902 |
| C07 BasicGJChronology.setYear(long,int) | 7 | 9 | 0.6 | 0.4 | 0.5389999999999999 | 12 | 0.0 | 1836 | 974 |
| C08 PrecalculatedZone.create(String,boolean,ArrayList<Transition>,DSTZone) | 8 | 13 | 0.2 | 0.8 | 0.46099999999999997 | 13 | 0.0 | 3816 | 1688 |
| C09 CachedDateTimeZone.nextTransition(long) | 9 | 3 | 0.8 | 0.2 | 0.632 | 14 | 0.0 | 1444 | 792 |
| C10 DateTimeZone.convertLocalToUTC(long,boolean) | 10 | 15 | 0.4 | 0.6 | 0.43699999999999994 | 5 | 0.8 | 2818 | 1333 |
| C11 DateTimeZoneBuilder.readFrom(DataInput,String) | 11 | 16 | 0.2 | 0.8 | 0.4175 | 15 | 0.0 | 1842 | 972 |
| C12 DateTimeZone.parseOffset(String) | 12 | 11 | 0.6 | 0.4 | 0.478 | 6 | 0.8 | 1738 | 900 |
| C13 AbstractInstant.toString() | 13 | 4 | 0.8 | 0.2 | 0.6080000000000001 | 16 | 0.0 | 1516 | 835 |
| C14 DateTimeZoneBuilder.toDateTimeZone(String,boolean) | 14 | 18 | 0.4 | 0.6 | 0.4 | 17 | 0.0 | 2824 | 1356 |
| C15 DateTimeZone.getDefaultProvider() | 15 | 20 | 0.2 | 0.8 | 0.3305 | 18 | 0.0 | 2098 | 1042 |
| C16 BaseDateTime.BaseDateTime(int,int,int,int,int,int,int,DateTimeZone) | 16 | 14 | 0.6 | 0.4 | 0.4475 | 7 | 0.8 | 1930 | 1017 |
| C17 DateTimeFormatter.parseLocalDateTime(String) | 17 | 5 | 0.8 | 0.2 | 0.5840000000000001 | 19 | 0.0 | 2270 | 1126 |
| C18 DateTimeZone.forID(String) | 18 | 21 | 0.4 | 0.6 | 0.32600000000000007 | 20 | 0.0 | 2106 | 1090 |
| C19 ISODateTimeFormat.dateTime() | 19 | 22 | 0.2 | 0.8 | 0.28700000000000003 | 21 | 0.0 | 1664 | 917 |
| C20 CachedDateTimeZone.createInfo(long) | 20 | 17 | 0.6 | 0.4 | 0.41700000000000004 | 8 | 0.8 | 1664 | 885 |
| C21 ISODateTimeFormat.tTime() | 21 | 10 | 0.8 | 0.2 | 0.536 | 22 | 0.0 | 1708 | 936 |
| C22 DateTimeZone.getOffsetFromLocal(long) | 22 | 23 | 0.4 | 0.6 | 0.252 | 9 | 0.8 | 2716 | 1386 |
| C23 ISODateTimeFormat.time() | 23 | 19 | 0.6 | 0.4 | 0.356 | 23 | 0.0 | 1704 | 925 |
| C24 Fraction.printTo(StringBuffer,long,Chronology,int,DateTimeZone,Locale) | 24 | 24 | 0.2 | 0.8 | 0.1565 | 24 | 0.0 | 1596 | 845 |

## 7. Reproduction

Use `python -m rave_fl.research_pipeline --output results/rave_fl_v3/reproduction_001 --report-dir reports`. This never invokes a model. See [the complete reproducibility guide](RAVE_FL_Reproducibility.md) for dependencies, test logs, resource blockers, strict reservation assumptions, and independent-bug commands. Historical ledgers are preserved and hashed. All plots have PNG and editable vector SVG versions in [figures](figures/), with source/model/budget metadata in [the figure manifest](tables/figure_manifest.csv).

