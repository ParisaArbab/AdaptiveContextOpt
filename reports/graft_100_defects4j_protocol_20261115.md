# GRAFT-FL: 100 real Defects4J bugs, preselected, batched and audited

Date: 2026-11-15
Branch: graft-fl-aiscalar-20261113
Status at protocol creation: **code prepared on GitHub, no 100-bug experiment executed**

## Question

Does inspecting graph-based code evidence improve **file-level bug localization**
compared with L1-only, and can test-aware selection or stopping reduce graph
calls, verification tokens and provider usage without losing accuracy?

The earlier three-bug development experiment found:
- L1-only ranked an official changed file first on Lang-1, Math-1, Time-1.
- Two real graph inspections per graph policy/fusion were performed.
- No file MRR improvement or test-aware stopping benefit was observed.
- Exact method-level fault labels were unavailable.
- The graph condition uses hashed, line-numbered excerpts of the complete
  observed failing test and stack trace, while retaining the complete source
  evidence separately. It is not the old full-log graph prompt condition.

## Frozen 100-bug sampling design

Official source: the **locally installed Defects4J** CLI, queried with
"defects4j bids -p <project>" on AIScalar. The exact Defects4J Git SHA,
active ID lists, deterministic selection SHA, bug IDs, and split are recorded
in the frozen cohort.json and committed **before executing these 100 bugs**.

- 20 each from Chart, Closure, Lang, Math and Time, exactly 100 distinct IDs
- Seed 42, independent deterministic sampling within each project
- Previously inspected Time-1, Time-4, Lang-1 and Math-1 excluded
- Batch order: 20 batches, 5 bugs per batch (one per project)
- First four batches (20 bugs): declared development set
- Remaining 16 batches (80 bugs): declared evaluation set
- No resampling if a checkout, complete test, error trace, build, BM25 index,
  graph package, or provider call fails
- This is a *frozen stratified sampling protocol*, not selection for difficult
  bugs after inspecting gold. Evaluate difficulty via the frozen L1 baseline
  after inference. The evaluation split is kept untouched until reporting.

## Evidence, labels and direct experimental comparison

Each bug is checked out at its authentic buggy revision with Defects4J.
The *first official Defects4J tests.trigger method* is selected from metadata
**before executing that test**, and `defects4j test -t Class::method` runs it
against the actual buggy checkout. Only its genuinely observed failing output
is accepted. This is a declared single-trigger-test condition for 100 bugs;
the earlier three-bug run used the full developer test suite and the two
conditions must not be silently treated as identical.

The preparer obtains the exact executed test method source, full observed
failing_tests log for that one test, source SHA values, actual checkout Git
revision, code-root metadata and modified classes. No fixed production source,
oracle patch, gold files or modified class names are added to L1/model-facing
failure prompts.

A bug is marked unavailable if it lacks the validated inputs. No synthetic
failures or reconstructed test descriptions may replace authentic observations.
Collection uses the original 9,000 byte upper bound *per complete test source
and per complete raw trace*, with no hidden truncation.

The candidate pool comes from 24 BM25 method candidates under the same
source and preprocessing rules. Every comparator uses this same pool.

Policies: L1-only anchor, L1+graph-focused, L1+test-aware-no-stop,
L1+test-aware-with-stop. Fusions: RRF and suspicion-rank blend.

Model: Qwen2.5:7b via local Ollama (temperature 0, seed 0,
8,192-token context, 160 output tokens, batch size 1).
Graph evidence budget: 4,096 byte-plus-template proxy units.
Per-policy logical budget: 90,000 units.
Maximum graph calls: 2 per graph policy.
No model is trained, fine-tuned or claimed calibrated in this comparison.

Capped real inference per five-bug batch:
- Up to 220 new provider requests
- Up to 500,000 physical provider input+output tokens across the batch
- Up to 100,000 physical provider tokens per bug
- Up to 20 such independent batches, so maximum **4,400 new requests**
  and **10,000,000 provider tokens** if every batch spends its full cap
- Exact within-run request cache reuse is not a new physical provider call.
  Cross-bug caching is not assumed.
- GPU and host compute time are not inferred from token counts.

These are worst-case cap totals, **not forecasts or actual measured usage**.

## Pre-inference safety checks

Before each real Qwen batch:
1. Match exact frozen IDs and split; verify all five official bug records exist.
2. Check reproducibility hashes, real evidence files and file-proxy labels.
3. Run GRAFT's own candidate/graph preflight; require 24 valid candidates
   and no missing or substituted bugs.
4. Check at least one real graph package in the top five BM25 candidates
   fits the unchanged 4,096-unit graph budget for each focused and test-aware
   selector. This feasibility check uses only observed tests and buggy code,
   not gold or hidden model outcomes.
5. Start real Qwen inference only on explicit --allow-inference permission.
6. If a batch stops, preserve real costs and partial results, mark it
   incomplete and continue the remaining predeclared batches without swapping
   bugs.

The pilot three-bug code is not overwritten. All 100-bug artifacts are stored
under data/graft_100_cohort_20261115 in separate batch directories. Git pushes
include frozen IDs, evidence, logs, audit, caches and accounting, but **exclude
large third-party source checkout directories**.

## Report honestly

Each five-bug batch is scored as complete only if all bug/fusion/policy
pairs complete, real focused/no-stop graph calls occur, and physical provider
usage is known, pending-free and non-synthetic. Unavailable or incomplete bugs
are never silently counted as correct, silently dropped, or replaced.

Aggregate over frozen development and evaluation sets separately, for each
fusion:
- Number predeclared, number authentically prepared, number fully paired
- Baseline and graph file-level Top-1, Top-5 and MRR
- Paired difference in file MRR between L1 and each graph policy
- Explicit exploratory subgroups based on **baseline L1 Top-1 hit versus
  L1 Top-1 miss**, without changing the original frozen 100 IDs
- Graph call count and whether stopping saved real graph calls
- Provider physical calls and input/output tokens (not additive logical costs)
- Missing bug IDs, real errors and cohort attrition, including project effects

Interpretation caveats: Defects4J classes.modified labels are *modified-file
proxies*, not verified exact faulty methods. MRR 1.0 for the first modified
source file does not prove exact bug location. Missing bugs reduce the usable
sample; a completed 100-ID selection is not automatically 100 successful pairs.
Changing the test/prompt/selection protocol later must be documented as a
new condition and not silently mixed into the fixed evaluation set.

## Run on AIScalar

    cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
    conda activate rave-fl
    git fetch origin graft-fl-aiscalar-20261113
    git merge --ff-only FETCH_HEAD

    # Runs offline regression checks, freezes IDs and pushes selection hash
    bash scripts/aiscalar_graft_100.sh freeze

    # Real Defects4J test collection and bounded local Qwen on all batches
    # after freeze. Detached to survive an SSH disconnect.
    bash scripts/aiscalar_graft_100.sh launch-all

Follow the log path printed by the launcher. Read cohort progress by:

    bash scripts/aiscalar_graft_100.sh status

For a conservative initial validation, use:
    
    bash scripts/aiscalar_graft_100.sh inspect 0
    bash scripts/aiscalar_graft_100.sh collect 0
    python -m scripts.graft_100_run preflight \
      --root data/graft_100_cohort_20261115 --batch 0

No Qwen inference is performed by freeze, inspect, collect, or preflight.
After the cohort completes or stops, the final report is:

    data/graft_100_cohort_20261115/aggregate/aggregate.md

## Evidence of execution

Only real AIScalar output and the archived GitHub result manifests can
establish actual Qwen costs, accuracy and completion. The initial code
commits do **not** establish results for these 100 bugs.
