# GRAFT-FL phased development

Work is opt-in on the existing GRAFT-FL branch. Historical requests, ranks and default RAVE-FL behavior are preserved. Each numbered phase has its own implementation/test commit and a decision record.

0. Versioned strict evaluation and frozen baselines.
1. Typed, provenance-backed evidence graph.
2. Relevance-only selection and complete-prompt budgeting.
3. TopoCompress-inspired dense/sparse, acceleration, propagation and redundancy selection.
4. Path augmentation and budgeted connectivity protection.
5. Observable dependency closure and fixed-budget repair.
6. Explicit sketch/graph/method-full observations and a bounded router.
7. Reproducible fixtures, archive diagnostics, provider-independent resource profiles and held-out experiment protocol.

The user will execute real model tests later, including on an A100. No live inference or automatic model/repository downloads are part of this development run. Published QA gains are not fault-localization measurements. Code completion and fixture success are not empirical localization gates. Advanced components remain opt-in until independent quality evidence exists.
