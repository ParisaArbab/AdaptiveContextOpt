# RAVE-FL: Risk-Aware Verification and Evidence Allocation

This is an **initial MVP**, not a validated fault-localization result.
It implements the inference-time decision structure in the RAVE-FL diagram
without changing the existing AdaptiveContextOpt benchmark pipeline.

## Architecture

```text
Bug report + stack trace + repository + failing test
                     |
           Base fault localizer
        (e.g. SieveFL, external)
                     |
         Ranked candidate functions
                     |
       Features + L0/L1/L2 sketches
                     |
       Rank-risk estimator (heuristic)
                     |
          Token-budget policy
                     |
         +-----------+----------+
         |           |          |
    BATCH_SCREEN  FULL_VERIFY  STOP
         |           |
         +-- verifier + token accounting
                     |
              feedback ledger
                     |
              final re-ranking
```

The proposed `EXPAND_CONTEXT` action is used as a cheaper L1 fallback
when full L2 verification does not fit. The full diagram's iterative
feedback-driven evidence expansion and calibrated decision policy
are **future work**.

## MVP API

- `Candidate`: a ranked function and base-localizer scores.
- `features`: semantic/coverage/complexity/calls/cost/rank-gap metadata.
- `sketch`: three evidence levels (L0 signature, L1 + calls, L2 + body).
- `heuristic_rank_risk`: placeholder sigmoid score, **not** a trained
  probability of ranking error and **not** calibrated.
- `RaveFL`: budget-aware choice of batch screening vs full verification;
  captures verdict, explanation, uncertainty, estimated input-token cost,
  and verifier-reported token cost; re-ranks inspected functions.
- `Verifier`: injected callable accepting sketches and action; use a real
  model adapter for experimental runs. Default is a non-informative stub.

## Try it

```bash
python -m rave_fl.demo
python -m unittest discover -s tests -p 'test_rave_fl.py'
```

The demo uses **synthetic scores and a mock verifier**, so its output
must not be reported as measured localization accuracy or efficiency.

## Next experiments

1. Adapt SieveFL output to `Candidate`, logging its initial ranking,
   stack-trace evidence, structural features and actual prompt token counts.
2. Collect held-out rank inversion / harmful-error labels with train/test
   separation by project and bug. Ground truth is for training and evaluation,
   never used in an inference-time decision.
3. Train a LightGBM risk classifier; assess calibration with Brier score,
   reliability plots, and expected calibration error.
4. Implement an LLM verifier producing bounded JSON verdict, explanation,
   uncertainty and actual API token counts. Limit output tokens *before*
   model calls, not only after.
5. Compare fixed full verification, budgeted random screening, heuristic
   RAVE-FL, and learned RAVE-FL on matched bug instances.
6. Measure Top-1, Top-5, MRR, recall, total input/output tokens, latency
   and cost; report paired bootstrap intervals across bugs.

## Research constraints

Avoid gold bug-label leakage. Baseline and RAVE-FL must receive the same
candidate set. The base-localizer ranking and any unseen candidate keep
their original relative order unless verified evidence changes the ranking.
Token estimation is character-based and should be replaced by a real tokenizer.
This is a research prototype, not the production-ready architecture.
