# Phase 4 — Bounded path preservation

Implemented multi-source, hop-bounded shortest paths over observed and explicitly static typed evidence links, followed by greedy loss-per-serialized-unit pruning. PART_OF containment is traversable in either direction; semantic similarity is never a trusted program path. Fixed terminals are selected from cheap scores, never gold labels. The protected set may be infeasible; unreachable targets remain explicitly reported. Connectivity is not causality.

The shared renderer now includes selected REQUIRES, MAY_CALL, STACK_AT and coverage links, and frame references include their failure-text line numbers. Their complete serialization cost is charged to every selector. Phase 2 measurements retain their earlier renderer and are not silently overwritten.

Validation: 26 cumulative GRAFT tests passed. On the 97 archived inputs, selecting the first prior candidate with a 4,096-byte-proxy input allowance gave 93 initially feasible packages. Path processing retained 90 feasible packages, made three protected sets infeasible, and retained four pre-existing infeasible inputs; 31 graphs had at least one reachable requested terminal. This is an offline structural diagnostic, not a localization experiment. Data: results/graft_fl/phase04/paths.json. Real model tokens are unavailable.

Decision: keep path preservation optional. Sparse observed links and infeasible protected sets rule out presenting this as guaranteed complete evidence. Continue with dependency closure to prevent selected fragments losing their observable prerequisites.
