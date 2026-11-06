# AITECH pilot protocol: dependency-preserving evidence selection

## Objective and hypotheses

**RQ1:** Does preserving evidence dependencies improve function localization
over relevance-only selection at identical token budgets?

**RQ2:** What is the compute cost of forming and preserving dependency closure?

H1: Dependency-aware selection improves gold-evidence recall and downstream
function localization on cases where relevance-only selection dangles.
H2: The improvement is not uniform, so a learned controller could avoid
unnecessary dependency expansion on easy cases. Both are untested.

## Unit of analysis

A *bug instance* with a repository revision, issue text, error logs,
failing tests, candidate evidence fragments, and an independent gold patch.

Each evidence record stores:
`{id, source_path, source_start, source_end, kind, text, tokens, score}`.

Each dependency record stores:
`{dependent_id, support_id, relation, confidence, provenance}`.

Initial relation types: `stack_frame`, `test_calls`, `imports`,
`call`, `defines`, `uses`. Keep source locations so that every
restored link is auditable. Do not infer causal links from co-occurrence alone.

## Conditions

A: RAW, full original failure evidence.
B: Relevance-only top-fragment selection.
C: Dependency-aware selection using explicit links.
D: C + on-demand ARISE slices.
E: C + learned missing-support classifier (once labeled data exists).
F: AITECH policy choosing B/C/D/E per instance (after supervised pilot).

Use identical candidate pools and downstream LLM/prompts for B and C.
For C and D, include graph build and retrieval costs in the reported budget
and latency. Run matched budgets (e.g., keep 30%, 50%, 70% of raw tokens);
use one tokenizer across conditions.

## Data partitions and leakage prevention

Split by repository, not by individual test failure or evidence fragment.
Never use gold patches, bug-fix commits, or support annotations during
inference. Record the base commit SHA before retrieving source code.
All model selection happens on development data only; report test once.

## Evaluation and controls

- File/Function Recall@1, @3; Function MRR; optional Line Recall@k.
- Gold evidence recall measured offline, with annotation provenance.
- Dependency dangling: retained fragments missing one or more *known*
  supports, plus missed dependencies from a small manual audit.
- Total LLM input+output tokens, retrieval tokens, agent tool calls,
  wall-clock latency, GPU time, and cost per successful localization.
- Separate graph construction/caching and per-instance overhead.
- Paired fix/break counts and confidence intervals across independent bugs.
- Compare equal token budget *and* equal wall-clock budget where possible.

## First acceptance gates

1. Synthetic example confirms that relevance-only selection can dangle.
2. Dependency-aware selection produces zero known dangling edges and stays
   within its budget.
3. Real-bug pilot includes evidence IDs from actual logs and code.
4. At least one isolated ablation measures whether dependency-aware
   restoration changes downstream function ranking.
5. No claim of generalization before independent repository-level testing.

## Reproduction logs

Record: `instance_id`, repository/base SHA, policy name, evidence hash,
dependency source, model ID, tokenizer ID, prompts, token counts,
runtime, predicted functions, ground-truth functions (scoring only),
and any tool errors. Save all intermediate selected and rejected IDs.

## Known limitations

A completeness score based only on known edges can miss real dependencies.
ARISE's current slicing does not cross procedure boundaries, and the
Referential Dangling pretrained classifier was trained on QA, not debugging.
A deterministic closure policy is a baseline, not itself learned AI.
