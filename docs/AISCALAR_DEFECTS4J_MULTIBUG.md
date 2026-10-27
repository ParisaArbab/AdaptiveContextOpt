# Run RAVE-FL V2 Utility on multiple real Defects4J bugs (AIScalar)

This procedure **does not** run the old compression benchmark. It uses the
`rave-fl-adaptive-pilot` branch and the real `scripts/rave_fl_run.sh` evaluator.

## Scope and safety

- One independently checked-out **buggy** Defects4J revision per bug.
- Real `defects4j test` output for observed failure evidence; no synthetic logs.
- Real SBIR, Ochiai and BoostN ranks from
  [FlexFL_OriginalReplication](https://github.com/ParisaArbab/FlexFL_OriginalReplication).
  A separate **sparse checkout** fetches only requested ranking CSVs and the
  offline ground-truth label file; it does not modify `references/FlexFL_Time25`.
- Gold labels are consumed only for offline scoring, not online candidate selection.
- Each bug has its own budget, journal, logs, reports and results. A failure or
  unknown token usage for one bug cannot block later bugs.
- Ollama should already be running on the dedicated authorized GPU endpoint.
  The scripts do not start, stop, or reassign another user's model server.
- The evaluator measures fault localization, not patch generation or bug repair.

## 1. Configure AIScalar

```bash
conda activate rave-fl
cd ~/RAVE_FL_Research/AdaptiveContextOpt_AdaptivePilot
git pull --ff-only origin rave-fl-adaptive-pilot
export PATH="$HOME/defects4j/framework/bin:$PATH"
defects4j info -p Time -b 24
curl --fail --max-time 5 http://127.0.0.1:11437/api/version
nvidia-smi -i 0
```

## 2. Prepare a first independent five-bug batch

```bash
mkdir -p logs
nohup python -u scripts/aiscalar_prepare_d4j_batch.py \
  --bugs Time-24 Time-26 Lang-1 Math-1 Chart-1 \
  --manifest configs/aiscalar_multi_d4j.json \
  --test-timeout 900 \
  > logs/multi_d4j_prepare.log 2>&1 < /dev/null &
echo "Preparation PID: $!"
```

Check `tail -n 30 logs/multi_d4j_prepare.log`. Some older Defects4J
projects require different Java/tool dependencies. Missing rankings,
checkout errors, timeouts, or nonreproducible failures are **explicitly skipped**
and logged. Do not count them as successful bugs.

Only bugs marked `ready` are included in `configs/aiscalar_multi_d4j.json`.
The raw observed outputs remain in each `checkout/failing_tests`; the compact
online evidence is separately stored under `data/rave_fl_observed_failures/`.
The chosen bug list is declared before looking at ground truth.

## 3. Preflight and prepare before spending tokens

```bash
bash scripts/rave_fl_run.sh preflight \
  --dataset defects4j \
  --dataset-file configs/aiscalar_multi_d4j.json \
  --candidate-provider flexfl \
  --candidate-count 24 --top-per-method 10 \
  --backend ollama --model qwen2.5:7b \
  --endpoint http://127.0.0.1:11437 \
  --device a100 --require-observed-failure --check-provider \
  > logs/multi_d4j_preflight.json

bash scripts/rave_fl_run.sh prepare \
  --dataset defects4j \
  --dataset-file configs/aiscalar_multi_d4j.json \
  --candidate-provider flexfl \
  --candidate-count 24 --top-per-method 10 \
  --require-observed-failure \
  --output-dir results/multi_d4j_prepare_01
```

Inspect the `statuses` in preparation output. If any case fails source-span
validation, inspect it before inferencing. A `prepare` run does not use the LLM.

## 4. Evaluate each prepared bug independently

Only after prep and GPU/provider checks pass:

```bash
mkdir -p logs
nohup bash scripts/aiscalar_run_d4j_batch.sh configs/aiscalar_multi_d4j.json \
  > logs/multi_d4j_experiment.log 2>&1 < /dev/null &
echo "Batch PID: $!"
```

The runner uses the Time-25-successful settings: Qwen2.5 7B, context
window 8192, batch size 2, Full + Fixed L1 + V2 Utility, max 2 full repairs
under V2, 900-second per-request timeout, strict admission and bounded
per-bug inference. Individual run directories are under
`results/d4j_multibug/<timestamp>/<bug>/` with reports, tables and ledgers.
The script logs exit status for each bug; it continues to the next even if
one bug fails. Use `tail -f logs/multi_d4j_experiment.log` for progress.

Do not compare aggregated **physical** tokens against a sum of per-policy
accounted costs; identical request-cache reuse changes total physical usage.

## 5. Report

For a completed run, read `results.json`, `summary.json` and
`reports/evaluation.md`. Report paired per-bug Top-1, Top-5, MRR, first
relevant rank, input+output tokens, full inspections, wall time and failures.
Do not discard unsuccessful bugs silently; show attempted vs completed
denominators, and count all selected independent bugs. Record whether ground
truth falls in the candidate pool. Run at least three bugs before interpreting
aggregate performance, then expand across projects after validating setup.
