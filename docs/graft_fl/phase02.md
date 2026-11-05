# Phase 2 - Relevance baseline and complete-message budgets

Added deterministic BM25 evidence selection, source-order JSON rendering, mandatory failure/target anchors, whole-statement packing, duplicate frame handling and accounting of the system message plus serialization. Missing targets and infeasible mandatory context fail explicitly. Graph hashes/indices are cached for repeated candidate queries.

Counters are explicit: the no-dependency default is a conservative UTF-8 content-byte proxy plus a 256-unit template allowance, NOT measured model tokens. A local-only chat-template tokenizer adapter is available without loading a model or downloading files. Provider reservation remains a separate boundary; tokenizer/model compatibility must be validated.

Archive packing diagnostic (first prior-ranked candidate per bug, 97 inputs): proxy budgets 2,048 / 4,096 / 8,192 admit 81 / 93 / 96 complete mandatory contexts. The remaining 16 / 4 / 1 are reported infeasible, never silently truncated. This is packing feasibility, not localization accuracy or token savings. Sixteen cumulative focused tests pass.

Decision: expose all three packing budgets; use 4,096 proxy units as a conservative development configuration and keep infeasibility visible. Real token comparisons require the target tokenizer. Proceed to graph propagation as an opt-in alternative while holding query/rendering/counter fixed.
