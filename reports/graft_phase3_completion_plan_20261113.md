# GRAFT Phase 3, completing evaluation without fabricated evidence

**Prepared:** 2026-11-13, `graft-fl-aiscalar-20261113`  
**Code status:** Collection, configuration, audit, and runner code committed to GitHub. **Not executed on AIScalar yet.**

## Why new work is needed

Phase 3 passed 47 tests and ran Qwen on one repeatedly studied bug, Time-4. The final rank did not improve with test-aware retrieval. The stopping rule had an open evidence question and did not activate. Separately, the frozen Time-1/Lang-1/Math-1 FlexFL manifest had **no bug reports, trigger-test files, or buggy checkouts** in the tested paths. Those missing inputs cannot be invented or silently replaced.

The following are **two separate, transparent follow-ups**:

### Condition A: Existing Time-4 with complete authentic test method

`scripts/prepare_graft_time4_authentic_test.py` reads the original saved Time-4 buggy checkout and the actual complete `TestPartial_Basics.testWith3` Java method via tree-sitter, preserving the original trigger description. It checks that Git HEAD is exactly the previously saved revision and tracked source is clean. It records SHA-256 for the test source/method, evidence, original trigger and revision. It never reads model outcomes or gold.

**This is not a newly observed test run.** The evidence kind remains `trigger_description`, not `observed_failure`, because the original failing trace was supplied rather than generated in this preparation.

```bash
cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
conda activate rave-fl
git fetch origin graft-fl-aiscalar-20261113
git merge --ff-only FETCH_HEAD
bash scripts/aiscalar_graft_phase3_time4_full_test.sh
```

The script runs regression tests, prepares a **new** config/manifest (does not alter originals), strictly checks preflight, then makes real bounded Qwen calls, and archives outputs. Qwen cap: 45 new calls, 60,000 known provider tokens.

Compare these results with the earlier `graft_phase3_test_aware_time4_20261113_210525` run. This tests whether authentic complete `catch`/exception expectation clarifies reasoning, but is development-only, not independent validation.

### Condition B: Predeclared three-bug exploratory study using authentic Defects4J

**Frozen IDs preserved:** `Time-1`, `Lang-1`, `Math-1`. Their original commit SHA values and selection rule are retained in `configs/rave_independent_smoke.json`.

The new collection condition uses Defects4J's official `checkout -p Time -v 1b -w ...` and `defects4j test -w ...` functionality. It collects:
- exact official buggy checkout and new actual Git SHA;
- actual `failing_tests` from running developer-written tests;
- **complete raw Java source for the executed failing test method**, not a fabricated or truncated snippet;
- exact original source paths, content SHA-256, test metadata and test stdout/stderr;
- official modified-class **file** proxy labels, clearly distinguished from true method-level gold.

The new dataset is `data/graft_phase3_official_<timestamp>/manifest.json`. It is created **only when all three cases succeed**, and it leaves the frozen old manifest untouched. The script refuses to overwrite existing source trees, truncate evidence, or accept dirty/mismatched checkouts. A new official Defects4J checkout may not equal the previously declared original revision; actual vs expected SHAs are recorded and the condition is explicitly different, never silently considered an exact replication.

**Important:** this is **BM25 candidate retrieval**, because the user's local FlexFL traditional ranking CSVs and trigger text were missing. The same BM25 candidate pool is used for all arms within each bug. **This is not the historical FlexFL condition.** The file-level modified-class proxy can support file-level Top-1/Top-5/MRR; it cannot justify claims about exact method-level bug localization. `candidate_pool_coverage` is reported so missing a bug in the top-24 pool cannot be mistaken for a bad re-ranker.

#### Step B1, inspect available dependencies with no checkout or test calls

```bash
bash scripts/aiscalar_graft_phase3_prepare.sh inspect
```

The earlier AIScalar preflight showed `defects4j=null`, so a fresh local CLI installation or PATH correction will probably be needed. Follow the official Defects4J README: https://github.com/rjust/defects4j. A typical installation requires Perl modules, `./init.sh`, and adding `framework/bin` to PATH. **Do not infer Defects4J is installed from an existing Python environment alone.**

#### Step B2, explicitly create authentic inputs and actually run tests

After `defects4j info -p Time -b 1` succeeds:

```bash
bash scripts/aiscalar_graft_phase3_prepare.sh collect
```

This executes real local checkouts and tests, and writes the per-bug provenance. If any bug fails, the collection report says `incomplete` and **does not create a runnable, silently shrunken cohort manifest**. Logs remain local for recovery; no experiment runs.

#### Step B3, run paired bounded Qwen comparison only with verified three-bug manifest

```bash
bash scripts/aiscalar_graft_phase3_multibug.sh \
  data/graft_phase3_official_<timestamp>/manifest.json
```

Replace `<timestamp>` with the actual preparation directory, printed by Step B2. The runner first runs the new tests, uses only the original three IDs, validates all prepared source/candidates/evidence, then runs real inference with maximum 135 new calls and 240,000 shared provider tokens. It compares:

| Policy | Test-aware source | Conservative stop |
|---|---|---|
| `l1_anchor` | no graph | no |
| `l1_graph_focused` | no | no |
| `l1_graph_test_aware_nostop` | yes | no |
| `l1_graph_test_aware` | yes | yes |

Each is measured with both `rrf` and `suspicion_rank_blend`. The audit `scripts/audit_graft_phase3_multibug.py` reports paired file MRR/Top-5, file-label candidate coverage, uncertainty guard use, graph calls, stop eligibility, and logical tokens. Physical provider calls/tokens are reported separately because L1 and graph requests may be shared via cache.

**Stopping rule evidence:** If the rule never activates on the real bugs, record exactly that, not a fake savings claim. The historical synthetic fixture already verified implementation behavior, not real-world benefit.

## Research integrity and limitations

- Observed failures are produced by real Defects4J tests in the new condition, never from gold or post-fix patches.
- Official modified classes provide file-level *proxy* labels. Method-level accuracy is **unavailable** until authentic independent method gold labels are supplied.
- The test-aware heuristic has not been calibrated; suspicion is not a trustworthy probability.
- These three IDs were studied before and are **exploratory**, not clean held-out evaluation.
- Defects4J, official checkout/source metadata and build compatibility must be available on AIScalar; these tools cannot execute AIScalar jobs remotely.
- If Java dependency/build incompatibility, missing test source, dirty checkout, or cost cap makes a bug unavailable, stop and retain diagnosis. Never substitute another bug after viewing results.
