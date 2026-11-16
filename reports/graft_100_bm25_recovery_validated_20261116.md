# BM25 corrected recovery, validated preflight

## Original results remain frozen

- Cohort: 100 predeclared Defects4J bugs, frozen selection SHA256 `a27224d014f0fa8c4c9ca431e3cd9e264342be1c58e40ce0024392df33c486b6`.
- Original 100-bug run: 58 fully paired (14 development, 44 held-out evaluation), 42 incomplete.
- Corrected code: BM25 and lexical retrieval keep the top-scoring occurrence of every distinct method ID, then take the first 24 unique IDs. The frozen 24-candidate quota has not changed.
- AIScalar validation supplied by researcher: six regression tests passed. Corrected 24-of-24 unique selection on Closure-32, Closure-106, Closure-170, Closure-85, Closure-161, and Math-58.
- AIScalar full GRAFT preflight: **6/6 batches READY**, **30/30 frozen bugs READY**, on batches 02, 06, 07, 11, 14, 19, each with 24 candidates. No Qwen inference was performed in these checks.
- These successes are **preflight readiness**, not completed experiment results. Do not add 30 to the original 58 until the actual model run and paired audit finishes.

## Isolated recovery procedure

The helper `scripts/graft_100_recovery_workspace.py` creates `results/graft_100_bm25_recovery_v1`. It copies the immutable cohort selection and uses symlinks to each original batch's already-authentic **prepared** folder, while model output, caches, configs, audits and ledgers are written only into the separate recovery root.

```bash
cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
conda activate rave-fl
git pull --ff-only origin graft-fl-recovery-analysis-20261116
python -m pytest -q tests/test_graft_100_bm25_dedup.py tests/test_graft_100_recovery_workspace.py
python -m scripts.graft_100_recovery_workspace

# Explicit Qwen inference, first try one development batch:
python -m scripts.graft_100_run run --root results/graft_100_bm25_recovery_v1 --batch 2 --allow-inference
```

The `run` command retains the original per-batch token and request caps, 24 BM25 candidates, frozen 5 bugs, graph policy settings, and two fusion methods, but writes a new implementation-versioned result. **It spends real provider tokens.** Check `results/graft_100_bm25_recovery_v1/batches/batch_02/experiment/completion_gate.json` for `completed: true` before attempting the other five batches. Incomplete results remain auditable. Do not run the original `scripts/aiscalar_graft_100.sh`, it is designed for the original branch and original root.

If batch 02 succeeds, run batches 06, 07, 11, 14, and 19 one at a time using the same command with the new batch index; do not overwrite existing outputs or change config to force a pass.

## Comparability limits

- The original run and this corrected-BM25 recovery have different candidate-pool implementation versions; they must be labeled accordingly.
- The corrected retrieval can change candidate rankings and model prompts. Recovered bugs are not a direct controlled estimate of graph-vs-baseline effects on the same population as the original 58.
- The original 58 remain as a fixed historical report. A combined descriptive sample, if produced later, must report counts by implementation version and avoid claiming a single uniform held-out experiment.
- Remaining failures outside these six batches include three test-evidence preparation defects and two incomplete inference cases. They require separate investigation.
