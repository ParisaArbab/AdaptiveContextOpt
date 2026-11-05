# Phase 5 — Observable support closure

Added transitive REQUIRES/PART_OF closure, cycle-safe dependency discovery and fixed-input-budget repair. When added prerequisites overflow the allowance, optional fragments and their dependent fragments are evicted together, using lost cheap relevance per saved serialized unit. Mandatory anchors and paths protected by Phase 4 remain protected. Their closure can be infeasible and is reported. This is a rule-based adaptation of referential repair, not the paper's trained detector, and never guarantees semantic completeness or resolves absent callees.

Coverage can now connect failure to an observed test only when the literal failing TEST identity exactly matches the supplied, checksummed instrumentation fact. Ranking scores never create coverage. No coverage was supplied for the archive diagnostic.

Validation: 30 cumulative GRAFT tests passed. With the final renderer and a 4,096-byte-proxy allowance for the first prior candidate, 97 archived inputs had 259 missing observable support links across 62 packages. Closure repair left zero missing links, with 93 feasible and four already infeasible packages. The input allowance was unchanged; optional evidence can be lost. Artifact: results/graft_fl/phase05/closure.json. No model calls or localization gains were measured.

Decision: expose repair as an explicit ablation and use it for structurally supported packages; retain rejection details. The remaining question is whether the changed evidence helps real judgments, which requires new prompts/responses. Next implement separate evidence stages and a bounded experimental router without replacing legacy controllers.
