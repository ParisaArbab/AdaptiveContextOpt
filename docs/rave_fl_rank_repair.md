# L1-first budgeted rank repair

The next experiment builds on the measured fixed-L1 baseline. It leaves
`shared_prior_evidence_fusion_v1`, full prompts, L1 prompts, model settings and
candidate pools unchanged. The controller screens the entire pool in the same
L1 batches, then spends the remaining budget on a small number of independent
Full calls. It never runs L0 or pays for an intermediate L1 expansion again.

The selection protocol is saved in `configs/rave_rank_repair_protocol.json`.
Time-25 was already inspected during development. Neither replay nor new
inference on that bug is an unseen-bug validation result.

## The four controls

- `repair_top`: verify the highest current fused-ranked unverified candidate.
- `repair_disagreement`: verify maximum SBIR/Ochiai/BoostN rank disagreement;
  break ties with the current ranking.
- `repair_suspicion`: verify the highest raw L1 suspicion. This simple additional
  control tests whether a more complicated utility is actually necessary.
- `repair_utility`: predict ranking movement from replacing a candidate's fused
  score by its raw L1 suspicion, score the positive movement of L1-derived
  beliefs with reciprocal-rank discounts inside Top-5, and divide by estimated
  additional Full tokens. Recompute after every measured Full result. Stop
  when no admissible action has positive predicted value.

For cheap evidence, the belief proxy is
`(1 - uncertainty) * suspicion + uncertainty * prior_score`.
For already Full-verified candidates it is the observed Full suspicion.
The prior score and final ranker are the same as the previous pilot. This proxy
is neither calibrated fault probability nor a learned expected MRR estimator.
A nonpositive proxy is an engineering stop rule, not proof that stopping is safe.
Invalid cheap evidence is prioritized for recovery if a Full call fits. Full
calls are capped (`--max-full-repairs`, default 4); every paid result still
changes the ranking, whether or not it helps offline accuracy.

Policies see only source, base ranks, L1/current observations and preflight
cost estimates. They never see gold, unselected Full responses or the actual
future output length. Candidates remain in the pool even if Full is unaffordable.
Admission remains approximate; actual overruns are charged and exposed. The
controller does not alter batch composition to make a new unseen group fit a
smaller replay budget: incomplete initial coverage is explicitly reported.

## Zero-inference replay

```bash
python -m rave_fl.replay \
  --run results/rave_fl_pilot/time25_llama31_8b/run.json \
  --manifest configs/rave_pilot_time25.json \
  --protocol configs/rave_rank_repair_protocol.json \
  --output results/rave_fl_rank_repair/replay_new
```

Replay requires complete measured Full and fixed-L1 references. It verifies
source/evidence hashes, all request-response pairs, exact independent messages,
model weight digest, generation settings, batch composition, actual provider
counters, raw/parsed observations and frozen ranking scores/tie-breaking.
Unknown actions or changed prompts fail; they do not trigger a model call or
invent missing outcomes. Planning reconstructs the original estimate and does
not use a future response's actual output token count. Replies are revealed only
when `execute()` is called for the chosen action.

`replay.json` records `new_llm_calls=0` and `fresh_inference_tokens=0`. Its accounted
tokens are the costs of recorded actions in a conditional replay, not a new API
bill. The replay assumes that reordering these history-free requests preserves
their observed responses. New inference is needed to check that assumption.
Gold is read only after all policy executions finish.

The offline oracle enumerates feasible subsets of zero through four Full calls,
using gold and observed future costs, to show the opportunity ceiling and
nonadditive pair effects. It is explicitly a diagnostic, never a selectable
online policy. Do not compare it to real policies as though it were deployable.

## Marginal-action training data

After replay, `action_values.jsonl` records candidate-specific changes in MRR,
Top-1 and Top-5, and the measured extra Full cost. It evaluates each action from
the initial L1 state and from distinct partial-repair states reached by the
primary policies. Other candidates' current evidence is held fixed. This makes
state-dependent action interactions visible rather than assuming that individual
improvements add up. Future Full observations and gold appear only in targets,
not in the inference feature map. Actual future token counts are targets;
preflight cost estimates are features.

Repeated rows/states from Time-25 remain **one independent bug**. No real learned
risk/utility artifact is trained from them. Collect independent bugs and split by
bug (preferably with repository holdout) before fitting or calibrating any model.
The earlier classification-risk training route remains separate from this
marginal-action-value regression dataset; their targets are not interchangeable.

## Fresh model experiment

```bash
python -m rave_fl.pilot \
  --manifest configs/rave_pilot_time25.json \
  --model llama3.1:8b --arms fixed_batch,repair_utility \
  --token-budget 16000 --max-full-repairs 4 --max-output-tokens 160 \
  --output results/rave_fl_rank_repair/live_original_new
```

A complete Full reference can be requested in the same invocation using
`--arms full,fixed_batch,repair_utility --token-budget 60000 --policy-budget 16000`.
Using the local installed Llama tag is an additional experiment, not a
reproduction of the supplied Qwen2.5/A100 report.

## Candidate-name ablation

```bash
python -m rave_fl.evidence_ablation \
  --manifest configs/rave_pilot_time25.json \
  --output-directory results/rave_fl_rank_repair/masked_new

python -m rave_fl.pilot \
  --manifest results/rave_fl_rank_repair/masked_new/manifest.json \
  --model llama3.1:8b --arms full,fixed_batch,repair_utility \
  --token-budget 60000 --policy-budget 16000 \
  --max-full-repairs 4 --max-output-tokens 160 \
  --output results/rave_fl_rank_repair/live_masked_new
```

The transformation uses **every candidate method name**, not the known fault.
It masks occurrences inside TEST/frame method identifiers with stable opaque
aliases. Assertion lines, expected/actual values, scenario suffixes, file/class
names, line numbers and frame repetition remain. The mapping is not sent to the
model. This removes a direct method-name cue, not all possible localization cues.
The metadata records all replacements, including generic candidate names such
as `test`; it is not a targeted substitution of only the gold method name.
This is a different evidence condition on the same bug, not a second independent
sample. Its Full reference must be run using that same masked evidence.

## Validation and research boundary

```bash
python -m pytest tests/test_rave_fl*.py -q
python -m rave_fl.audit_pilot \
  --run results/rave_fl_rank_repair/live_original_new/run.json \
  --manifest configs/rave_pilot_time25.json \
  --output results/rave_fl_rank_repair/live_original_new/audit.json
```

The current policies are untrained development heuristics. Matching a strong
simple heuristic is not evidence of novelty. In particular, compare the
single-Full raw-suspicion baseline, not just a baseline forced to spend four
calls. On multiple independent bugs, preregister acceptance margins, compare
paired MRR/Top-1/5 and costs, then report uncertainty at the bug level. The
candidate-name ablation and repeated calls do not increase independent n.

Related work already includes efficiency-guided learned code-localization
controllers: [FuseSearch](https://aclanthology.org/2026.findings-acl.143/) uses SFT
and RL for adaptive parallel tool execution. This pilot instead investigates
post-localizer evidence allocation/rank repair. That distinction alone does not
establish a novel contribution; a wider literature and empirical comparison is
still needed before making a paper claim.

## Explicit post-hoc numeric follow-up

The naming ablation exposed a failure even in Full verification. One further
variant was therefore specified *after seeing that result*, with no policy or
model changes. `--normalize-assertions` mechanically extracts unambiguous UTC
offsets from the original JUnit expected/actual fragments, spells out their
numeric difference, and retains all the raw masked assertions. It never
reconstructs ellipsized dates or adds a diagnosis/function identity. Unsupported
formats fail rather than inventing values. This is not held-out validation:

```bash
python -m rave_fl.evidence_ablation \
  --manifest configs/rave_pilot_time25.json --normalize-assertions \
  --output-directory results/rave_fl_rank_repair/normalized_masked_new
```

The exploratory extension and stopping decision are saved in
`configs/rave_rank_repair_posthoc_evidence.json`. Adding normalized text changes
both presentation and repetition/length; do not claim that this isolates only
one semantic factor. No further variants are searched after this one.
