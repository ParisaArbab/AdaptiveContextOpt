# GRAFT-FL, 100 predeclared Defects4J bugs

Only real observed failure inputs and file-level modified-class proxy gold.
No outcomes are fabricated. Incomplete and unavailable bugs are excluded
from paired means and counted separately.

Selection SHA256: a27224d014f0fa8c4c9ca431e3cd9e264342be1c58e40ce0024392df33c486b6
New physical Qwen calls across archived batches: 1691
Physical provider input+output tokens across batches: 2532709

| Split | Fusion | Predeclared | Fully paired with real graph | L1 MRR | Focused MRR | Test-aware MRR |
|---|---|---:|---:|---:|---:|---:|
| development | rrf | 20 | 14 | 0.5675 | 0.5675 | 0.5675 |
| development | suspicion_rank_blend | 20 | 14 | 0.6888 | 0.6888 | 0.6888 |
| evaluation | rrf | 80 | 44 | 0.4948 | 0.4948 | 0.4948 |
| evaluation | suspicion_rank_blend | 80 | 44 | 0.5320 | 0.5320 | 0.5320 |

## Exploratory subgroup by baseline L1 Top-1 (not used for sampling)

| Split | Fusion | L1 baseline result | Fully paired n | Mean L1 MRR | Mean focused MRR |
|---|---|---|---:|---:|---:|
| development | rrf | l1_top1_hit | 7 | 1.0000 | 1.0000 |
| development | rrf | l1_top1_miss | 7 | 0.1349 | 0.1349 |
| development | suspicion_rank_blend | l1_top1_hit | 9 | 1.0000 | 1.0000 |
| development | suspicion_rank_blend | l1_top1_miss | 5 | 0.1286 | 0.1286 |
| evaluation | rrf | l1_top1_hit | 15 | 1.0000 | 1.0000 |
| evaluation | rrf | l1_top1_miss | 29 | 0.2334 | 0.2334 |
| evaluation | suspicion_rank_blend | l1_top1_hit | 16 | 1.0000 | 1.0000 |
| evaluation | suspicion_rank_blend | l1_top1_miss | 28 | 0.2646 | 0.2646 |

The evaluation split (80 frozen bugs) is not a guarantee of 80 completed
runs. Top-1, method-level ground truth and latency require separate
grounded reporting, not interpretation of file-level MRR.
