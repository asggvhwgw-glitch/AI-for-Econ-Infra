from __future__ import annotations

from itertools import combinations
import numpy as np

from ...compute.clusters import normalize_cluster_arrays
from ...errors import InputError, error_boundary
from .results import ClusterDiagnostics, ClusterDimensionDiagnostics


def _cluster_scores(result, state, codes: np.ndarray) -> np.ndarray | None:
    if getattr(result, "estimator", None) != "ols":
        return None
    X = np.asarray(state.X_within, dtype=np.float64)
    u = np.asarray(result.residuals, dtype=np.float64)
    if X.ndim != 2 or len(X) != len(u):
        return None
    w = getattr(state, "weights", None)
    base = u if w is None else u * np.asarray(w, dtype=np.float64)
    G = int(codes.max()) + 1
    scores = np.empty((G, X.shape[1]), dtype=np.float64)
    for j in range(X.shape[1]):
        scores[:, j] = np.bincount(codes, weights=X[:, j] * base, minlength=G)
    return scores


@error_boundary("cluster_diagnostics")
def cluster_diagnostics(result, *, include_pairwise_intersections: bool = True) -> ClusterDiagnostics:
    """Report cluster balance and score concentration on the estimation sample.

    Flags are review diagnostics, not rules for choosing a clustering level.
    ``few_clusters`` uses G<30; balance/concentration flags are explicitly
    heuristic and should be interpreted with the empirical design.
    """
    state = getattr(result, "state", None)
    if state is None:
        raise InputError(
            "cluster diagnostics require keep_state=True",
            code="inference.missing_state", stage="inference",
            suggestion="Refit with keep_state=True to retain estimation-sample cluster codes.",
        )
    clusters = normalize_cluster_arrays(state.clusters, nobs=len(state.y_within), required=True)
    n = len(clusters[0])
    dims: list[ClusterDimensionDiagnostics] = []
    for j, c in enumerate(clusters):
        G = int(c.max()) + 1
        sizes = np.bincount(c, minlength=G).astype(np.float64)
        shares = sizes / float(sizes.sum())
        effective = 1.0 / float(np.sum(shares * shares))
        mean = float(np.mean(sizes))
        cv = float(np.std(sizes) / mean) if mean > 0 else np.nan
        score_share_max = None
        score_effective = None
        scores = _cluster_scores(result, state, c)
        if scores is not None:
            norms = np.linalg.norm(scores, axis=1)
            total = float(norms.sum())
            if total > 0:
                score_shares = norms / total
                score_share_max = float(np.max(score_shares))
                score_effective = float(1.0 / np.sum(score_shares * score_shares))
        flags: list[str] = []
        if G < 30:
            flags.append("few_clusters")
        if float(np.max(shares)) > 0.10:
            flags.append("dominant_cluster_size_heuristic")
        if cv > 1.0:
            flags.append("unbalanced_cluster_sizes_heuristic")
        if score_share_max is not None and score_share_max > 0.20:
            flags.append("dominant_cluster_score_heuristic")
        dims.append(ClusterDimensionDiagnostics(
            dimension=j, n_clusters=G,
            min_size=int(np.min(sizes)), median_size=float(np.median(sizes)),
            mean_size=mean, max_size=int(np.max(sizes)), size_cv=cv,
            largest_share=float(np.max(shares)), effective_clusters_size=effective,
            singleton_clusters=int(np.sum(sizes == 1)),
            score_share_max=score_share_max, score_effective_clusters=score_effective,
            flags=tuple(flags),
        ))

    intersections: dict[tuple[int, int], int] = {}
    if include_pairwise_intersections and len(clusters) > 1:
        for a, b in combinations(range(len(clusters)), 2):
            # Dense pair factorization without an object/string interaction.
            ca = clusters[a].astype(np.int64, copy=False)
            cb = clusters[b].astype(np.int64, copy=False)
            base = int(cb.max()) + 1
            intersections[(a, b)] = int(np.unique(ca * base + cb).size)
    return ClusterDiagnostics(nobs=n, dimensions=tuple(dims), pairwise_intersections=intersections)
