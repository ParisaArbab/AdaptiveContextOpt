# RAVE-FL V3 Reproducibility and Artifact Guide

## One-command offline regeneration

From the repository root, with Python 3.11 and the project's existing dependencies:

```bash
python3 -m rave_fl.research_pipeline \
  --output results/rave_fl_v3/reproduction_001 \
  --report-dir reports
```

Choose a **new** `--output` path for every diagnostic execution. The entry point does not call a model or download code. It validates saved paired/smoke ledgers, records historical hashes and environment/test logs, evaluates six controllers under both admission modes, exports CSVs, and regenerates the English Markdown reports plus PNG/SVG figures. It preserves raw historical results. `--report-dir` contains generated review artifacts and may be regenerated from the saved diagnostic dataset.

Expected required Python imports: `pytest`, `numpy`, `matplotlib`, `tree_sitter`, `tree_sitter_java`, `lightgbm`, `sklearn`. Existing LightGBM synthetic tests also need its platform OpenMP runtime. Exact installed versions and executed command vectors are saved by the entry point under `provenance/`. This task uses the existing environment rather than installing new model weights or Java toolchains.

Individual stages:

```bash
python3 -m rave_fl.v3_diagnostics \
  --protocol configs/rave_v3_protocol.json \
  --output results/rave_fl_v3/diagnostics_reproduction_001

python3 -m rave_fl.research_report \
  --diagnostics results/rave_fl_v3/diagnostics_reproduction_001 \
  --output reports

python3 -m pytest -q tests/test_rave_fl*.py
python3 -m pytest -q tests \
  --ignore=tests/test_pipeline_dependencies.py \
  --ignore=tests/test_agent4sr_format_finalization.py

# Run known failures explicitly; they remain visible, not silently fixed.
python3 -m pytest -q tests/test_pipeline_dependencies.py \
  tests/test_agent4sr_format_finalization.py
```

The report-only command expects audit/test provenance under the report directory from the pipeline. Reuse the generated directory when rendering again. Dataset fingerprints include source/evidence/ledger hashes, protocol hashes, model digest/settings and implementation hashes. Git HEAD alone is insufficient to identify uncommitted research code; hashes and starting-state records address that limitation. Final generated-file hashes are saved separately, avoiding a circular self-hash.

## Current executed tests

{{TESTS}}

The three unrelated failures are two dependency tests referencing absent `scripts/run_pipeline.sh`, and one Agent4SR finalization test expecting a missing seen entity. Source files are preserved. The initial RAVE subset also exposed a pre-existing uncommitted replay identity regression test; V3 fixes that failure rather than excluding it. Earlier V1/V2 test counts are historical reports, not newly added tests.

## Data contracts and token accounting

- `candidates.csv`: one row per candidate/evidence condition, initial and L1 ranks/scores, SBIR/Ochiai/BoostN top-list and full-CSV positions, raw BoostN scores (SBIR/Ochiai scores unavailable), censored disagreement, gaps, lexical support and source-risk features. Fields ending `_offline` contain future outcomes or gold interpretation; they must never become controller inputs.
- `policies.csv`: one row per policy/condition/budget/call-cap/admission combination. `complete_l1` distinguishes incomplete discovery. `tokens` are accounted historical provider costs, not fresh replay billing. `budget_exceeded` is explicit.
- `actions.csv`: pre/post ranking state and marginal MRR after each selected call. L1 batch costs are repeated for candidate-level diagnostic rows with `batch_cost_shared=True`; sum costs over unique call keys, not candidate rows.
- `decisions.csv`: every candidate option at complete-L1 decision points, predicted cost, available cheap features, selection/rejection/stop reason, current balance. Full outcomes are absent from this option feature audit.
- `pairs.csv`: all 276 two-candidate combinations per evidence condition. These are offline score-swap interventions, not fresh model interactions or independent bugs.
- `diagnostics.json.gz`: complete policy states, exact replayed calls, traces, reservation events and provenance.
- `figure_manifest.csv`: sources, condition, token definition, model, budget and interpretation limit for each exported plot.

Historic admission is estimated. Strict admission reserves prompt UTF-8 bytes plus pinned template overhead and capped output, bounded by context. This bound is specific to the frozen Llama 3 byte tokenizer/plain two-message template, not a universal tokenizer theorem for every model/backend. Strict runs check returned counters and abort on a violated assumption or unavailable response. Actual future costs are never consulted to admit a selected action.

The original and current source trees can differ; replay checks exact regenerated prompts and raw stored request fingerprints. A source change under the same ID now fails at execute time. A changed batch grouping, failure context, model setting or missing Full response is unsupported and must not be fabricated. Historical corpus validation requires complete Full/L1 references.

## Historical and saved-result inventory

- Phase A: `README.md` and `docs/rave_fl_phase3c_evidence_ablation.md`; Qwen original result JSONs are absent locally. The supplied PDF is a historical summary, not a substitute for missing raw ledgers.
- V1: `results/rave_fl_pilot/time25_llama31_8b/` and `time25_llama31_budget16k/`.
- V1 labels: `results/rave_fl_pilot/time25_rank_risk_dataset.jsonl`; one independent bug, no real trained/calibrated artifact.
- V2 fresh saved runs: `results/rave_fl_rank_repair/live_original/`, `live_masked/`, `live_normalized_masked/`.
- V2 replay/oracle/action labels: `results/rave_fl_rank_repair/replay_final/`, `replay_masked/`, `replay_normalized_masked/`.
- Mask transformations: `masked_evidence/` and `normalized_masked_evidence/` with manifest hashes and unchanged raw assertion text.
- Pre-existing uncommitted controls/smoke: `results/rave_fl_iteration_d/`; starting-state file hashes and audited smoke results identify their provenance. Not newly executed in V3.
- Interrupted/obsolete artifacts remain untouched, including an earlier unknown-usage preflight. They are excluded from completed-arm comparisons, and no whole-workspace billing total is claimed.

## Independent-bug commands and prerequisites

V3 resource preflight selected Time-2, Lang-2 and Math-2 before any new outputs; no Gold files for those bugs were used to choose them. `configs/rave_v3_independent_protocol.json` predeclares the 12-bug exploratory set. Defects4J is not on PATH, and the three new official buggy checkouts plus `failing_tests` are unavailable. A Java 17 installation exists, but project-specific build/JDK compatibility is unverified. Therefore no new verified-test inference was performed.

Once an official Defects4J installation and a compatible JDK are available, run the following from the repository root. This prepares *buggy* checkouts into new directories; it does not patch them to fixed revisions. Required SBIR/Ochiai/BoostN input CSVs must separately be populated from the same ranking source using the existing sparse-data workflow; mismatches remain errors.

```bash
command -v defects4j
java -version
mkdir -p logs/rave_v3_independent

for bug in Time-2 Lang-2 Math-2; do
  project=${bug%-*}
  number=${bug#*-}
  checkout="data/rave_fl_checkouts/$bug"
  test ! -e "$checkout" || exit 1
  defects4j checkout -p "$project" -v "${number}b" -w "$checkout" \
    > "logs/rave_v3_independent/${bug}_checkout.log" 2>&1 || continue
  git -C "$checkout" rev-parse HEAD \
    > "logs/rave_v3_independent/${bug}_revision.txt"
  defects4j test -w "$checkout" \
    > "logs/rave_v3_independent/${bug}_tests.log" 2>&1
  test -s "$checkout/failing_tests" || continue
  python3 -m rave_fl.failure_evidence \
    --failing-tests "$checkout/failing_tests" \
    --output "logs/rave_v3_independent/${bug}_evidence.txt"
done
```

Review build/test logs and confirm the observed failures come from the declared buggy revisions. Preserve raw `failing_tests`; official Defects4J instrumentation can alter build metadata, so record HEAD plus dirty state/hashes rather than concealing it. A missing/failed case remains in the attempted-bug manifest; do not replace it based on outcomes.

Create a live-compatible prepared manifest only after all prerequisites pass. This command does no inference and requires actual buggy failure evidence:

```bash
python3 - <<'PY'
import json, subprocess
from pathlib import Path
m = json.loads(Path('configs/rave_independent_smoke.json').read_text())
m['cases'] = []
m.pop('pre_inference_budget_note', None)
for bug in ['Time-2', 'Lang-2', 'Math-2']:
    checkout = Path('data/rave_fl_checkouts') / bug
    evidence = Path('logs/rave_v3_independent') / (bug + '_evidence.txt')
    if not (checkout/'.defects4j.config').exists() or not (checkout/'failing_tests').is_file() or not evidence.is_file():
        raise SystemExit('Incomplete verified buggy evidence for ' + bug)
    revision = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    m['cases'].append({'bug':bug,'flexfl_repo':'references/FlexFL_Time25',
                       'checkout':str(checkout),'buggy_revision':revision,'evidence_file':str(evidence)})
m['selection_rule'] = 'Preselected Time-2/Lang-2/Math-2; no outcome-based replacement'
m['failure_evidence'] = 'Mechanically compressed local official buggy failing_tests; retain raw logs'
m['policy_budget'] = 20000
m['admission_mode'] = 'strict_reservation'
m['reservation_specification'] = 'configs/rave_v3_token_reservation.json'
m['policies'] = ['fixed_l1','highest_suspicion','ranker_disagreement','ranking_ambiguity','v2_utility','uncertainty_aware']
Path('configs/rave_v3_independent_prepared.json').write_text(json.dumps(m,indent=2)+'\n')
PY

# Source/candidate/cost preflight only; inspect estimates and incompatibilities.
python3 -m rave_fl.independent_smoke \
  --manifest configs/rave_v3_independent_prepared.json \
  --output results/rave_fl_v3/independent_preflight_001

# Bounded live collection only after valid sources/evidence/model preflight.
python3 -m rave_fl.independent_smoke \
  --manifest configs/rave_v3_independent_prepared.json \
  --output results/rave_fl_v3/independent_live_001 --live
```

The physical cap is 80,000 provider tokens and 100 new calls, with whole-context-plus-output reservation before uncached requests. Per-policy strict reservation is separate. A complete 30-candidate × three-bug Full-plus-L1 reference could require 114 independent requests, exceeding that cap; incomplete collection is expected to stop honestly rather than silently extend a sweep. Exact within-run selected-action cache reuse avoids repeated independent calls across policies. Fixed model digest/settings and original batch composition are required for later offline budget/call-cap replay. No complete 12-bug inference sweep is initiated by these commands.

Potential instrumentation/build failures, missing ranking CSVs, span mismatches, candidate-pool recall failures and incomplete global-cap executions must all be retained. A saved trigger description without locally executed evidence is a different, explicitly labeled evidence condition. Do not mix it with verified buggy-test results.

## Git branch and preservation

The task began on `feature/rave-fl-mvp` at the same commit as `codex/rave-fl-adaptive-pilot` (`635a316`). The latter local branch was checked out and renamed to `rave-fl-adaptive-pilot`, then published with upstream `origin/rave-fl-adaptive-pilot`. No force-push or history rewrite was used, and uncommitted starting files were preserved.

Authenticated GitHub API checks found no open pull requests and no protection on `codex/rave-fl-adaptive-pilot`. The repository has two collaborators; remote metadata cannot establish whether the other collaborator uses that old branch locally. **The old remote branch is retained.** A final fetch revealed 32 additional collaborator commits through `e6e9fd9` not included in the evaluated branch. Review and coordinate integration of those commits before deletion; the exact list is in `provenance/parallel_remote_branch.json`. Coordinate migration with the other collaborator/repository owner before deleting `origin/codex/rave-fl-adaptive-pilot`; this is the exact remaining safety blocker. The branch rename/publish is complete, and development continues on the requested new name.

Unrelated untracked `.tools/`, `data/repos/`, and `tests/test_pipeline_dependencies.py` remain outside the research commit. The pre-existing untracked analysis files included in the completed research change are identified in the starting-state provenance record. Historical results and manifest hashes are checked for preservation before final commit.
