# GRAFT Phase 3: repair of AIScalar offline test failures (2026-11-13)

## User-observed failing command

`bash scripts/aiscalar_graft_test_aware_offline.sh`, followed by a direct attempt to run `bash scripts/aiscalar_graft_test_aware_live.sh`. The offline suite reported **42 passed, 5 failed, 1 setup error**, therefore no real Phase-3 Qwen inference ran.

The first failures were infrastructure/API integration errors and **must not be interpreted as fault-localization accuracy measurements**.

## Causes and changes

1. **Wrong Origin API, two places.** `rave_fl.graft.graph.Origin` exposes `path`, `start_line`, `end_line`, `source_hash`, and `extractor`, **not** `provenance`. Both `rave_fl/graft/test_aware.py::failure_source` and `rave_fl/graft/selection.py::render_focused` used `failures[0].origin.provenance`. Now they read `failures[0].origin.extractor`. This field correctly preserves `supplied_text:trigger_description` provenance and does not change the source evidence or historical graph policies.
2. **Pytest miscollected selector as test.** `tests/test_graft_test_aware.py` imports the callable `test_aware_select` as a module-level name beginning with `test_`; pytest attempted to invoke it with a non-existent `graph` fixture. The function now sets `test_aware_select.__test__ = False` to prevent collection; real test functions still invoke the selector.
3. **Missing temporary SQLite directory in new integration test.** The new stopping-ablation test passed two uncreated `tmp_path /...` subdirectories into `test_graft_controller.setup`. The existing `Store` opens SQLite directly in the supplied directory, so the test now creates each isolated directory before constructing the store. The production Store is deliberately not changed.

## Verification status

These corrections were committed on `graft-fl-aiscalar-20261113`. **No new AIScalar tests were executed by the assistant**, and no Phase-3 Qwen experiment has been performed yet. The user should rerun:

```bash
cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
conda activate rave-fl
git fetch origin graft-fl-aiscalar-20261113
git merge --ff-only FETCH_HEAD
bash scripts/aiscalar_graft_test_aware_offline.sh
```

The script must finish its pytest run and save its zero-inference prompt audit before the real inference script should be run:

```bash
bash scripts/aiscalar_graft_test_aware_live.sh
```

If any new tests fail, **stop before Qwen inference** and capture full pytest output. Never change gold labels, source revisions, or historical comparisons to make a test pass.
