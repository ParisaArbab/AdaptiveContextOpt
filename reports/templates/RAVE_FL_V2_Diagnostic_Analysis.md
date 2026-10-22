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

{{GOLD}}

{{PROXY}}

Raw input ranker diagnostics (deeper ranks are not used by the evaluated top-ten controller features):

{{RANKSIGNALS}}

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

{{TIES}}

C22's Full suspicion remains 0.8. Its worse Full-all rank is exactly explained by changed competitors under the shared sort rule: the earlier constructor/test/conversion/base-date candidates receive equal or higher scores. The candidate table gives every affected method. This is not simply an allocation failure, because Full-all inspects every method yet still loses Top-5 under masked evidence.

### Oracle opportunity versus an online-selectable action

Masked one-Full oracle replay upgrades C22 and reaches rank 1 at 12,999 tokens. It uses the future Full result and gold labels to pick the action; this is an offline diagnostic ceiling. Highest raw L1 suspicion chooses a different, high-head-suspicion candidate and remains at rank 23. Disagreement also favors a different candidate; uncertainty-aware selection reaches only rank 22 with one call. All tested online policies miss the masked Top-5 opportunity through four upgrades. Exact budget/call-cap tables are in [the V3 report](RAVE_FL_V3_Experiment_Results.md).

An online policy could see C22's source sketch, low prior rank, 0.4 suspicion, 0.6 reported uncertainty, weak lexical support and ranker input positions. It could not know its unseen Full score 0.8 or gold identity. Oracle opportunity alone does not establish that a realizable low-cost policy can select that action reliably. Higher budget by itself does not repair Utility's zero-gain stop rule.

## 3. Numeric augmentation: explicitly post-hoc

After the masked failure, a deterministic parser **prepended** three assertion summaries to the complete masked evidence. Each states expected +04:00, observed +03:00, difference +3,600,000 milliseconds. It retains the raw ellipsized assertion text, does not reconstruct missing dates, and adds no candidate identity or diagnosis. This increases the context from 1,502 to 1,904 characters. The exact prompts remain in the two conditions' `_calls.jsonl` files and input snapshots.

The following counts compare all 24 candidate scores between masked and numeric conditions:

{{NUMERIC_CHANGES}}

The faulty method's L1 suspicion remains 0.4, uncertainty 0.6, fused score 0.252, and rank 23. Its Full suspicion remains 0.8. Numerical clarification changes some competing Full scores and wording but does not produce a stronger cheap signal for the gold method. Full-all worsens from rank 6 to 9; L1 and Utility remain rank 23; cost increases in every arm. The numeric gold Full costs 1,386 rather than 1,289 tokens, and the oracle mixed state costs 13,678. This is a failed post-hoc development intervention, not independent validation or evidence that numeric detail is generally harmful.

Short raw explanations illustrate the difference between an observed score and a plausible narrative:

{{RESPONSES}}

A more specific explanation in the numeric Full response does not improve its score or final position. Neither the ledger nor this controlled comparison proves that the model actually performs arithmetic correctly or grounds the offset change in the faulty branch. Prompt length, repeated numbers and altered evidence presentation also change together; do not infer a unique causal formatting mechanism.

## 4. Action interactions

{{ACTION_PROFILE}}

Positive actions can promote the gold method or demote competitors; negative actions can promote a competitor above it. Each marginal label is conditional on a recorded state, not a property of the method alone. Full inspection can therefore harm ranking. The strongest pair interaction in the original all-L1 starting state is:

{{PAIR}}

Let \(G(A)=MRR(A)-MRR(\varnothing)\). The interaction is

\[
I(i,j)=G(\{i,j\})-G(\{i\})-G(\{j\}).
\]

For the gold/conversion pair, the positive gold upgrade and negative competitor upgrade combine nonadditively. The original action-row dataset also changes C22's marginal MRR gain from 0.75 in all-L1 to 0 in a state where earlier distractor demotions already put it first. `convertLocalToUTC` ranges from −0.05 to −0.5 across recorded prefixes. Such changes make static independent-action utility unreliable.

The exhaustive original pair table contains {{BOTH_POSITIVE}} examples in which **both** singleton gains are strictly positive but their combination is net negative against all-L1. That stronger scenario is possible in richer ranking systems but is not demonstrated by this dataset. Do not conflate negative interaction with that claim. A future interaction-aware extension should begin with incumbent/tie/score-crossing features and sequential feature recomputation, not an unvalidated complex optimizer.

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

{{CANDIDATES}}

## 7. Reproduction

Use `python -m rave_fl.research_pipeline --output results/rave_fl_v3/reproduction_001 --report-dir reports`. This never invokes a model. See [the complete reproducibility guide](RAVE_FL_Reproducibility.md) for dependencies, test logs, resource blockers, strict reservation assumptions, and independent-bug commands. Historical ledgers are preserved and hashed. All plots have PNG and editable vector SVG versions in [figures](figures/), with source/model/budget metadata in [the figure manifest](tables/figure_manifest.csv).
