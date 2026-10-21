# Phase 2: real FlexFL inputs and an honest baseline

RAVE-FL can now read **existing real FlexFL method-ranking CSVs** for
Defects4J bugs and evaluate against the FlexFL ground-truth file. It does
not download the benchmark or silently fabricate missing files.

## Run on your computer / AIScalar

Clone the external FlexFL replication using the existing reference setup,
or point to an already cloned FlexFL tree:

```bash
./scripts/clone_reference_repos.sh
python -m rave_fl.experiment \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --bug Time-25 --method Ochiai --token-budget 2000 \
  --output results/rave_fl_phase2/time25_ochiai.json

python -m unittest discover -s tests -p 'test_rave_fl*.py'
```

The loader expects `data/FL_results/Ochiai/Defects4J/Time-25_method-susps.csv`
and `data/input/ground_truth/Defects4J/gt.json` inside the FlexFL tree.

## Interpretation

- Candidate IDs and ordering are from a **real external localizer file**.
- The interim semantic score is a monotonically decreasing **rank proxy**,
  not a semantic-retrieval score. Coverage is unspecified.
- The risk estimate is **untrained** and not a probability of true error.
- Source bodies are not yet connected, so L2 sketches can be just identifiers.
- No actual LLM verification is performed; baseline and RAVE-FL ranking
  metrics are intentionally identical.
- The estimated sketch token counts are **not** complete model usage, so
  do not interpret their difference as experimentally verified savings.

## Next integration requirement

Attach the actual buggy checkout's method bodies (prefer a method extractor
with stable line ranges), model verifier with provider token usage accounting,
and hard input/output token budgets. Then conduct a paired run on the same
Defects4J bugs with fixed model, prompts, candidates, and temperature,
comparing full verification against a learned risk-aware controller. Report
Top-1, Top-5, MRR, and actual input/output tokens and latency.

**Scientific rule:** ground truth must never be read inside the policy
or passed to the verifier; only the offline evaluation stage may use it.
