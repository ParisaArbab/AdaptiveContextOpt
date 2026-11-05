# GRAFT-FL development artifact index

Start with [the research progress report](RESEARCH_PROGRESS.md). It documents implemented modules, measured archive diagnostics, limitations, budget selection and the next independent-defect gate. [Verification](verification.json) records the executed test command, 188 passing tests, exact dependencies, synthetic-run accounting and source-code hashes. The 47 new GRAFT tests are listed in `validation/new_test_inventory.txt`; 141 other tests in the executed command existed before these phases.

Five figures are saved as both PNG and SVG in `figures/`; the underlying feasibility, cost and frozen ranking tables are in `tables/`. `summary.json` retains graph/cost counts and exact implementation hashes, while `provenance.json` identifies the 388 directly consumed archive files rechecked unchanged. This does not claim a new full hash audit of every file in the imported results tree.

`validation/fixture_run.json` and `fixture_resume.json` show two synthetic source fixtures, 14 complete arms, ten simulated requests and zero additional requests on resume. Physical model tokens were zero. Their persistent raw artifacts remain under `results/graft_fl/phase07_verified_fixture/`. Fixture ranking numbers are deliberately not promoted as research results.

`validation/a100_preflight.json` records the unavailable new smoke manifest and local NVIDIA hardware. The actual A100 server was not contacted. No new Defects4J inference, test execution or real BGE-M3 encoding is claimed. [The execution guide](../../docs/GRAFT_FL.md) explains how to prepare the later test and change models/budgets safely. The preselected exploratory list is Lang-54, Math-80 and Closure-61; these are not claimed as completed or held-out results.

Regenerate the report data into a new directory:

```bash
python3 -m rave_fl.graft.cli diagnostics --archive results-100-bugs/results/d4j_multibug/d4j97_promptfix_20261027_122406 --output /tmp/graft-diagnostics-new
python3 -m pytest -q tests/test_graft_*.py tests/test_rave_eval.py tests/test_rave_fl*.py
```

The graph JSON files and detailed first-target package rows are in `results/graft_fl/development_20261105/graphs/` and `structural.json`; they are regenerable outputs and not copies of new model responses. Historical source results, raw ledgers and old report artifacts were preserved. The report and phase documents identify measured structural changes separately from unproven model-quality hypotheses. See `docs/graft_fl/phase00.md` through `phase07.md` for the separate phase decisions and commits.
