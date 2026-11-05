# Phase 3 - TopoCompress-inspired selection

Implemented frozen dense/sparse relevance, query-gated second-order semantic acceleration, semantic k-nearest neighbors, source-order sequential edges, restart propagation, optional typed-program edges and redundancy-aware packing. Acceleration is computed inside each supplied method in source order, never in candidate rank order. Isolated nodes retain their mass; nonconvergence/invalid weights fail explicitly. Structural FL units and mandatory failure/target anchors make this an adaptation, not a claim to reproduce LongBench results.

The selector requires exact-text-hashed dense AND sparse vectors with a pinned model identity. Missing vectors and truncated inputs do not silently fall back to fake embeddings. The explicit `python -m rave_fl.graft.encode` exporter supports a complete local BGE-M3 checkpoint through optional FlagEmbedding/Transformers, with hub access disabled, truncation checks and model-file hashes. It was not executed with real weights in this development run. Fixture embeddings are distinctly marked synthetic; no learned or accuracy claim is made.

Validation: 22 cumulative focused tests, including mass conservation, bridge-score recovery, zero/isolated graphs, fixed-input propagation ablation, absent/truncated vectors and offline-only model prerequisites. Graph sources/diagnostics are immutable so cached hashes cannot drift through normal mapping mutation.

Decision: keep BM25 as the runnable no-model default while real dense/sparse artifacts are unavailable. Expose semantic and typed propagation as opt-in controls; do not enable them on the strength of synthetic scores. Proceed to bounded path preservation and dependency checks, keeping their empirical value unproven until the user's later real-model tests.

Primary method sources: Asante and Li, TopoCompress v2, https://arxiv.org/abs/2608.30811v2; supplied PDF `new papers/Asa26.pdf`. No paper performance figure is imported as a fault-localization result.
