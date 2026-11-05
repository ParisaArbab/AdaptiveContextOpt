# Phase 0 - Evaluation and frozen baselines

Added `java_strict_v2` without changing the historical matcher. Empty argument lists are explicit rather than wildcards; Java case is preserved; constructor markers, generic erasure, arrays/varargs and parameter declarations are handled. Conflicting qualified types are not merged. Unqualified/malformed method labels are unavailable, not invented zero accuracy. Labels supplied by an archive are not independently certified reference truth.

Validation: 7 focused tests. Offline re-evaluation covers 97 archived bugs; 84 jointly completed labeled bugs reproduce the legacy metrics exactly. Strict MRR: prior 0.517774, Full 0.485642, Fixed L1 0.386828, V2 0.393222. The known-labeled operational denominator includes 88 bugs, including failed labeled attempts. No raw archive file was written.

Decision: retain Prior as a mandatory baseline. Preserve historical fusion and use RRF as a separately named frozen/development comparison; do not attribute a fusion change to evidence selection. Proceed to evidence extraction; no claimed controller superiority or new inference.

```bash
python3 -m pytest -q tests/test_graft_evaluation.py
python3 -m rave_fl.graft.archive --source results-100-bugs/results/d4j_multibug/d4j97_promptfix_20261027_122406 --output /tmp/graft-phase00-new
```
