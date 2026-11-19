# New experiments: frozen Top-8 L1 + Code Cards and high-recall filtering

This branch contains two **different** research experiments. The first tests
whether the 10-bug development observation of identical L1 MRR with fewer
logical tokens generalizes. The second measures candidate-pool file recall
when selecting only 24 methods from a 384-method BM25 pool. Neither has new
results until it is executed on AIScalar.

## Experiment A, 15 independent development bugs, real Qwen

Fixed policy before collecting new data:

1. Retrieve original **24 BM25 methods** using the saved GRAFT bug settings.
2. For BM25 prior positions **1 to 8**, give Qwen the original L1 evidence.
3. For positions **9 to 24**, give Qwen a short deterministic Code Card.
4. Rank candidates by **0.30 * original prior + 0.70 * measured suspicion**,
   then deduplicate files in score order.
5. Compare with L1-for-all and Code-Cards-for-all using the **same candidates,
   same failure evidence, same Qwen2.5:7b model, seed 42, temperature 0,
   60,000 token budget, and 24-method pool**.

The policy is entirely fixed by BM25 rank, with no labels, future LLM scores,
or gold-based stopping. The code uses exact measured provider token counts.
**Only two types of independent observations, L1 and Card, are collected**,
instead of spending tokens on Full and dependency replies not used here.
Estimated paid Qwen calls: at most **15 x 24 x 2 = 720** successful calls,
plus at most one retry for any invalid reply. This count is a theoretical
maximum given 24 candidates per bug, not a promise of run time or cost.

The three additional five-bug groups are from the existing GRAFT 100-bug
cohort, and do not overlap with the 5 original pilot or 10 earlier development
bugs:

| Batch | Fixed bugs |
|---|---|
| 03 | Chart-19, Closure-4, Lang-62, Math-60, Time-5 |
| 05 | Chart-26, Closure-120, Lang-59, Math-54, Time-12 |
| 07 | Chart-10, Closure-170, Lang-20, Math-53, Time-10 |

The script refuses a changed protocol implementation after the first
`preflight`, a different observation manifest/model, changed batch IDs,
missing source checkouts, and any unapproved paid inference. It logs each
provider attempt and resumes from the saved per-batch ledger. Gold is imported
only for the offline `report` step. An interrupted checkpoint is preserved
and invalidates exact physical-token accounting as needed.

### Run on AIScalar

**AIScalar import fix:** Call the frozen study via `bash research_method_agent/run_mixed_top8_aiscalar.sh`. The wrapper sets `PYTHONPATH` to the repository root so Python can import the existing `rave_fl` package. The study source itself is deliberately unchanged, because the first (failed) preflight already saved its frozen SHA256 in `frozen_protocol.json`. The earlier `ModuleNotFoundError: No module named 'rave_fl'` occurred *before* model inference. You do not need to delete the protocol file or rerun the Qwen observations.

```bash
conda activate rave-fl
cd ~/RAVE_FL_Research/MethodDiscovery

git fetch origin research/mixed-top8-retrieval-20261119
git merge --ff-only FETCH_HEAD

# No Ollama calls, check all tests and materialize original buggy source inputs.
bash research_method_agent/run_aiscalar_sample.sh test
bash research_method_agent/run_mixed_top8_aiscalar.sh preflight

# Real, explicitly authorized Qwen calls. Each batch has a separate ledger.
# No gold labels used during these calls.
set -o pipefail
bash research_method_agent/run_mixed_top8_aiscalar.sh run-03 --allow-inference 2>&1 | tee mixed_top8_batch03.log
bash research_method_agent/run_mixed_top8_aiscalar.sh run-05 --allow-inference 2>&1 | tee mixed_top8_batch05.log
bash research_method_agent/run_mixed_top8_aiscalar.sh run-07 --allow-inference 2>&1 | tee mixed_top8_batch07.log

# Gold labels are used ONLY to evaluate the saved predictions.
bash research_method_agent/run_mixed_top8_aiscalar.sh report
```

Each batch defaults to no more than 300 new calls per run and 800,000
physical tokens total. If the collector pauses at a cap, review `status`
and decide whether to increase a limit before resuming that same batch:

```bash
bash research_method_agent/run_mixed_top8_aiscalar.sh status
bash research_method_agent/run_mixed_top8_aiscalar.sh run-03 --allow-inference --max-calls 400 --physical-token-cap 1000000
```

Use the retry example **only** when the ledger shows `collection_paused_at_cap`.
Provider exceptions with unknown token usage stop rather than retrying
silently.

The final paired report is:

- `results/mixed_top8_validation_15bugs/report.md`
- `results/mixed_top8_validation_15bugs/results.json`

A credible benefit is comparable MRR with a repeatable token reduction and
few paired regressions, with uncertainty reported. No success is guaranteed.
Because the original 10 bugs were used to choose Top-8, these 15 are a new
**development confirmation cohort**, not a final untouched test. Reserve
at least 30 different bugs for final, frozen evaluation.

## Experiment B, retrieve BM25 384 and keep only 24, NO Qwen

Run this separately, ideally first, to learn whether a cheaper retrieval
filter can recover the faulty file without increasing the expensive candidate
count. It uses only the already-studied ten bugs in GRAFT batches 01 and 02:

```bash
python research_method_agent/audit_high_recall_filter.py
```

It always retrieves the original **384 unique BM25 method candidates** from
buggy source, then selects at most 24 methods using one of five
*predeclared, gold-blind* rules:

- `bm25_top24`: original top 24 by BM25.
- `unique_file_first`: best BM25 method from each distinct file first.
- `two_per_file`: at most two methods per file before filling.
- `failure_anchor_first`: exact observed failure filename/method anchors then BM25 order.
- `failure_anchor_file_balanced`: failure anchors plus one candidate per distinct file.

It checks that the original top-24 order matches the saved 10-bug manifests.
**Only after selection** does it load modified-class gold to measure file
coverage. This is a retrieval-only upper bound, not LLM MRR or paid inference.

Output:

- `results/egav_validation_10bugs/high_recall_filter_audit.json`

We must inspect false negatives, gold-file candidate rank, and selected-file
coverage before choosing any retrieval heuristic. Even if 384-candidate
BM25 recall was 10/10 on this development cohort, a 24-method filter may
lose the correct file. Never interpret deep-pool recall as final localization.

## Reproducibility and limitations

- Do not compare historical GRAFT MRR numerically to this agent's MRR without
  matching model prompts, candidate sets, and fusion.
- `code_card` changes the code content **and** the `Evidence form` prompt label,
  so these are not isolated compression-only trials. A later controlled
  comparison must vary code while holding the prompt instructions fixed.
- The only gold is file-level modified-class proxy data, not curated faulty
  method labels.
- The method here is still fixed Top-8, not a learned risk predictor.
- The stored physical collection tokens include L1 and Card observations for
  all candidates, while reported policy tokens count only the selected
  observations and do not measure wall-clock policy latency.