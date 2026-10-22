# RAVE-FL flowchart-to-code alignment audit (2026-10-22)

**Reviewed branch:** `codex/rave-fl-adaptive-pilot` at `635a3167fd2f998e1c9668c028ac2a0b69b85f88`.

**Status: partial alignment.** The closed-loop verification MVP exists, but the repository does **not** yet implement or validate every method in the proposed flowchart. This is a source-and-reported-results review. This audit did not rerun local Ollama, Defects4J or the Python tests. Do not describe the architecture graphic as a fully completed implementation.

## Diagram versus executable source

| Flowchart component | Status | Where / precise boundary |
|---|---|---|
| Bug report, failing test, repository code | **Implemented, from prepared artifacts** | `rave_fl/compare_expanded.py:load_context`, `rave_fl/pilot.py:prepare_case`, `rave_fl/source_spans.py`; failure evidence and checked-out source are required. The pilot does not itself invoke Defects4J. |
| Base localizer (SieveFL example), semantic retrieval and runtime pruning | **Not integrated in this pilot** | `rave_fl/expanded_pool.py:union_pool` reads existing **SBIR/Ochiai/BoostN** CSV rankings. Separate `wp1/` modules implement parts of the broader Graphify/Agent4SR workflow, but are not called by this pilot as a live SieveFL localizer. |
| Ranked candidate function list | **Implemented with a different candidate source** | `union_pool` constructs the real ranked pool. `semantic_score` in this adapter is an *ordinal rank prior*, not measured semantic similarity; its `coverage_score` is zero, not actual coverage. |
| Low-cost features: ranks, agreement, complexity, cost, failure relevance | **Implemented with proxies** | `rave_fl/code_sketch.py:prepare_candidates/state_features`. Rank agreement comes from the traditional localizers; failure relevance uses lexical overlap, not learned relevance. |
| L0 signature, L1 structural sketch, L2 full code | **Implemented** | `rave_fl/code_sketch.py:java_sketch` uses tree-sitter-java, flags parse errors/truncation, and preserves L2 source. L1 is lossy, not equivalent to program slicing. |
| Rank-risk predictor | **Bootstrap implemented; learned path available, not trained on real independent bugs** | `rave_fl/risk.py`, `rave_fl/train_risk.py`. Bootstrap output is **not** a calibrated harmful-error probability. LightGBM and isotonic calibration require independent train/calibration/test bugs. |
| Adaptive token decision policy | **Heuristic implementation** | `rave_fl/adaptive.py:decide` chooses actions using remaining budget, estimated costs, risk, uncertainty, suspicion and disagreement. **No contextual bandit or reinforcement-learning policy** has been trained or evaluated. |
| Four actions: BATCH_SCREEN, EXPAND_CONTEXT, FULL_VERIFY, STOP | **Implemented** | `rave_fl/core.py:Action`, `rave_fl/adaptive.py:decide/execute`, `rave_fl/pilot_backend.py`. Expansion means L1 structural context; it is **not** a general targeted retriever for runtime/source evidence. |
| LLM execution, explanation, uncertainty | **Implemented** | `rave_fl/pilot_backend.py` sends real Ollama requests, validates JSON and records explanations, uncertainties, request/response ledgers and actual provider token counts. |
| State feedback and reconsideration | **Implemented** | `rave_fl/adaptive.py:execute` records observation history, promotes evidence levels monotonically, updates budget and reruns the decision policy after each call. |
| Final re-ranking, Top-1/Top-5/MRR | **Implemented using a pilot ranker** | `rave_fl/adaptive.py:fused_score/ranked`, `rave_fl/pilot_evaluation.py`. Fusion is deterministic, prior-preserving and **not the original SieveFL final LLM ranker**. |
| Token savings, latency and cost | **Partially implemented** | Verification input/output tokens and local LLM call durations are measured. End-to-end localizer cost, total pipeline latency and actual API dollar billing are **not** measured. |
| Initial MVP: baseline, sketches, risk, screen/full, compare | **Mixed** | Baseline and sketches and screen/full comparisons exist; an *offline* LightGBM trainer exists, but no genuinely held-out, real-trained risk model is supplied. |
| Rank-repair and strict offline replay | **Implemented follow-up beyond the flowchart** | `rave_fl/rank_repair.py`, `rave_fl/replay.py`, `rave_fl/rank_repair_labels.py`. Rank-repair utility is an **untrained proxy**, not a guaranteed MRR estimate. |

## Two distinct threshold concepts

- **0.60** is the *historical candidate-rank gate* in `rave_fl/adaptive.py` for the `rank_threshold` baseline. It inspects only candidates with ordinal rank prior >= 0.60; it is **not** a calibrated harmful-error probability and does not apply to the new adaptive screening policy.
- **0.35** is the default **bootstrap risk escalation threshold** (`PolicyConfig.risk_threshold`, CLI `--risk-threshold`). Bootstrap risk is an uncalibrated heuristic, so choosing 0.35 does not establish a statistical safety guarantee.
- In the `repair_utility` arm, the system uses neither as a learned accuracy guarantee; it screens L1 first, then upgrades candidates when its untrained positive ranking-utility proxy justifies the estimated token cost.

## What measured results actually show

From the checked-in reports, **one already inspected Defects4J bug (Time-25)**, not a general benchmark:
- Full / Llama3.1-8B: **21,378** provider tokens, buggy function **rank 2**.
- Fixed L1 batches: **11,298** provider tokens, bug **rank 4**.
- Budget-16k adaptive: **14,717** provider tokens, bug **rank 11**, outside Top-5.
- Fresh original-evidence rank repair: **12,516** provider tokens, bug **rank 1**.
- Under masked candidate names, fixed L1 and rank-repair both gave **rank 23**; name-independent localization is not demonstrated.

See `docs/rave_fl_pilot_results_2026_10_22.md` and `docs/rave_fl_rank_repair_results_2026_10_22.md` for the distinct runs and methodological limits. Do not mix historical Qwen2.5 totals with the newer Llama3.1 totals.

## Requirements before declaring *full* flowchart alignment

1. **Wire a live base localizer** into the pilot rather than only loading precomputed CSVs. Record the base localizer's real semantic/coverage scores and runtime-pruning provenance, without relabeling ordinal ranks as semantic or coverage measures.
2. **Gather independent bugs** and freeze a bug-level, preferably repository-level, train/calibration/test split. Train, calibrate and evaluate rank-risk using the existing `train_risk.py` tooling. Do not split multiple candidate rows from Time-25 across train and test.
3. **Evaluate the control policy honestly** against Full, fixed batching, and rank-repair under identical inputs. If a contextual bandit/RL is added, state its reward, state/action space, logged propensities or exploration design, and held-out evaluation. A weighted heuristic is not RL.
4. **Improve evidence grounding** under candidate-name masking and other cue ablations before optimizing allocation further. Test genuine targeted expansion of missing runtime/source/structure evidence if it is part of the intended diagram.
5. **Run multiple independent projects**; report paired Top-1, Top-5, MRR, candidate coverage, verification input/output tokens, end-to-end latency and cost separately. Check budget overruns, truncated prompts, invalid model replies, and stop safety. Do not claim accuracy preservation from a one-bug development result.
6. **Rerun code and environment tests** on the target compute platform. This document audits source and checked-in reports; it is not an independently executed test pass.

### Important implementation distinction

`rave_fl/core.py:RaveFL` is a **minimal earlier demonstration** with an injectable stub verifier. The executable measured closed-loop policy is `python -m rave_fl.pilot` backed by `rave_fl/adaptive.py:AdaptivePilot` and `rave_fl/pilot_backend.py:OllamaPilotBackend`. Do not treat the demonstration as proof of model-backed behavior.
