# Phase 1 · fixed-schedule comparison harness

12 November 2026. Baseline branch `GRAFT-FL` at `e64d6dee9e642adaf08e4485512e7060f3335403`. Phase 1 was explicitly approved. Both arms were exercised using synthetic vectors and replies.

## Implementation

- [Fixed-evidence runner](../../../rave_fl/graft/fixed_evidence_experiment.py) adds the opt-in module with `dry-run`, `run` and `resume`. It reuses `load_instances`, `prepare_one`, `build_graph`, `topo_select`, `GraphBackend`, `Store`, `Fusion` and the offline evaluator. The existing graph backend owns request formatting, admission and logical accounting. The new runner supplies the scheduled target and selection. The request identity also scopes journal keys to the fixed experiment and defect ID.
- [Example configuration](../../../configs/graft_fixed_evidence_dry_run.json) accepts local dataset, embedding and label paths, the model/provider settings, existing bug/project/limit filters and per-arm, per-bug and run caps. Paths in the example are placeholders for a prepared synthetic fixture. Configuration resolves resource paths locally and uses POSIX file locking already present in the project. Java and Python method fixtures were tested on macOS.
- [Schedule tests](../../../tests/test_graft_fixed_evidence_schedule.py) and [accounting tests](../../../tests/test_graft_fixed_evidence_accounting.py) cover the paired plan, source validation, label isolation, the real SQLite journal, invalid replies, caps and resume. A sentinel test fails if the fixed collector calls `Router.execute`.

Each bug is prepared once, with a stable prior-rank sort and input-order tie handling. The plan records both named topology configs and enforces their one-key difference. The same graph, embedding bundle, candidate identities, prompt template, allowance, model settings and RRF settings feed both arms. Paths, repair and typed propagation are disabled. Plan rows contain the exact requests and selected node IDs. The dry-run uses backend planning with an inert backend-kind record. It instantiates neither an execution provider nor a journal. Its output has no timestamps or run-directory-dependent identities.

Both arms receive the full scheduled target list. A context, source, package or resource infeasibility keeps the pair incomplete. A preflight failure blocks both arms for that defect before collection. The plan conservatively checks the full schedule against caps, including context plus output reservations for physical admission, and retains the reason. A paid invalid verdict stays in the journal and usage totals. Its pair is excluded from matched metrics even if the other arm completes. The online manifest rejects embedded labels and reference patches. An optional distinct label manifest with inline `gold` objects opens after collection. Matched results use strict Java and historical profiles for Defects4J, and the existing exact-ID legacy evaluator for the synthetic custom adapter.

The locked run directory holds the immutable plan and provider identity, the original Store journal and raw ledger, usage and checksum-bound results. Resume checks current preparation and resources against the saved plan and reconciles the journal with the raw request ledger before collection. A completed request can be replayed from the journal after an interruption. Pending, unknown and inconsistent records stop resume. The saved completed result is returned only after its checksum and its bindings to the manifest and journal verify. External shared-cache reuse is disabled for this phase. Exact in-run journal reuse is supported.

## Validation

| Check | Outcome |
|---|---|
| New focused tests | PASS. 14 cases within the final regression suite. |
| Legacy GRAFT and RAVE regressions | See final suite result in the phase log. |
| Two standalone dry-runs, Python hash seeds 7 and 19 | PASS. Exit 0 each. `cmp` exit 0. Both plans had SHA-256 `aca62957fa4adb98385c3dbdff0a59af6db5f28fd2d6e7168e012c14d383b511` and zero model calls. |
| Synthetic CLI collection | PASS. Java and Python both complete with four scheduled inspections each across the two arms. Eight simulated calls, 3,466 simulated input/output tokens, zero real inference tokens. |
| Synthetic CLI resume | PASS. Exit 0, identical usage, zero new simulated calls. |
| Scope and environment | Local macOS validation. Linux, real embeddings, target tokenizer parity and real model behavior remain unverified. |

The standalone check used `/tmp/graft-phase1-cli.jLs4c7/config.json`, the two `plan-final-*.json` files and `run-final/` under that directory. The fixture used `eval.fixtures.create` and the test-only `synthetic_bundle`. It split labels into their own file before planning. Command forms, run from the repository root with the prepared config, were:

```bash
scratch=/tmp/graft-phase1-cli.jLs4c7
phase_python="$HOME/.pyenv/versions/3.11.0/bin/python"
PYTHONHASHSEED=7 "$phase_python" -B -m rave_fl.graft.fixed_evidence_experiment dry-run --config "$scratch/config.json" --output "$scratch/plan-final-a.json"
PYTHONHASHSEED=19 "$phase_python" -B -m rave_fl.graft.fixed_evidence_experiment dry-run --config "$scratch/config.json" --output "$scratch/plan-final-b.json"
cmp "$scratch/plan-final-a.json" "$scratch/plan-final-b.json"
"$phase_python" -B -m rave_fl.graft.fixed_evidence_experiment run --config "$scratch/config.json" --output "$scratch/run-final"
"$phase_python" -B -m rave_fl.graft.fixed_evidence_experiment resume --config "$scratch/config.json" --output "$scratch/run-final"
```

| Standalone artifact | SHA-256 |
|---|---|
| Online fixture manifest | `6c2feca42de64e924888f41060fe6399a83170a25ea7b87f388c53e87acebbc2` |
| Run manifest | `b119c385c7307d40f9af0a9ceb5cf86473a5260aa2f816e43d877a8fb649628d` |
| Results | `f3d73e67133f1800ae8619f8d795bdd7d50aacc340e25379d78cc4543e9a467b` |

## Open gates

Phase 1 execution accepts only the fake backend. Real-model settings remain configurable in a dry plan. The current planner sums conservative whole-context physical reservations. A draft live allowance that fits after verified template-token accounting may still appear infeasible in this prototype. This is a budget-planning limitation, with no measured spend or quality implication. The existing immutable resume rule binds a run to its recorded implementation hashes, including any older run started before this module existed.

Phase 2 would verify the shared real embedding bundle, exact text and source provenance, and selector stability offline. The local BGE-M3 checkpoint and FlagEmbedding dependency were unavailable in Phase 0. Resolve that input gap before an actual encoder smoke. Target tokenizer equivalence, the three new bugs' checkout/evidence/candidate resources and server portability remain later gates.
