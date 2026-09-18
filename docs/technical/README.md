# Technical documentation and innovation policy

This directory contains mathematical/algorithmic documentation, implementation notes and the package-wide technical-innovation audit.

## Innovation source of truth

- `innovation-audit.md` — complete package audit distinguishing genuine technical innovation from established methods and engineering.
- `innovation-registry.json` — machine-readable release contract. Every item classified as `technical_innovation` must have a formal `.tex` manuscript and compiled `.pdf`, plus implementation/test/evidence mappings.

The current registered innovations are:

1. exact arbitrary-G categorical HDFE structural rank / absorbed DoF;
2. exact arbitrary-G numerical residual-core reduction;
3. exact partition-refinement HDFE canonicalization and structural design reduction.

No other current feature should be described as an original econhdfe technical contribution without updating the audit, literature boundary, registry and formal manuscript in the same release.

## Domain documentation

- `hdfe/` — exact rank/DoF mathematics, exact residual-core projection, solver implementation notes and exact-rank backends.
- `structural-design/` — exact partition-refinement / dependency-DAG design reduction manuscript.
- `cluster-inference.md` — established CRV/WCR cluster-inference behavior and support boundaries; this is technical documentation, not an originality claim.
- [`heterogeneous-specification-optimization.md`](heterogeneous-specification-optimization.md): internal acceleration for interaction-rich and heterogeneous-coefficient empirical specifications, including model integration and dense-fallback boundaries.
