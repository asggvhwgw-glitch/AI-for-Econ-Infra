from __future__ import annotations
from itertools import combinations
import numpy as np
from ...compute.wls import as_2d
import scipy.linalg as la
from scipy.stats import chi2, f
from ...compute.encoding import factorize_interaction
from ...compute.vcov import ols_vcov, score_covariance



def _residualize_small(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.shape[1] == 0:
        return A.copy()
    coef, *_ = la.lstsq(B, A, cond=None, lapack_driver="gelsy")
    return A - B @ coef


def _weighted_triplet(endog, exog, instruments, weights=None):
    E = np.asarray(endog, dtype=np.float64)
    if E.ndim == 1:
        E = E[:, None]
    n = len(E)
    C = as_2d(exog, n)
    Z = as_2d(instruments, n)
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
        E, C, Z = E * sw, C * sw, Z * sw
    return E, C, Z


def _first_stage_test(dep, C, Z, *, vce, clusters, df_absorbed, nested_adj,
                      time=None, panel=None, bandwidth=None, kernel="bartlett", df1_override=None,
                      effective_n=None, score_scale=None):
    n = len(dep)
    n_eff = float(n if effective_n is None else effective_n)
    zp = _residualize_small(Z, C)
    ep = _residualize_small(np.asarray(dep)[:, None], C).ravel()
    q = zp.shape[1]
    if q == 0:
        return {"partial_r2": np.nan, "f_classic": np.nan, "f_classic_pvalue": np.nan,
                "f_robust": np.nan, "f_robust_chi2_pvalue": np.nan, "excluded_instruments": 0}
    b, *_ = la.lstsq(zp, ep, cond=None, lapack_driver="gelsy")
    fit = zp @ b
    resid = ep - fit
    tss = float(ep @ ep)
    rss = float(resid @ resid)
    r2p = 0.0 if tss <= 0 else max(0.0, min(1.0, 1.0 - rss / tss))
    df1 = int(df1_override or q)
    df2 = max(n_eff - C.shape[1] - q - int(df_absorbed), 1)
    f_classic = ((tss-rss) / max(df1, 1)) / (rss / df2) if rss > 0 else np.inf
    p_classic = float(f.sf(f_classic, df1, df2)) if np.isfinite(f_classic) else 0.0
    bread = np.linalg.pinv(zp.T @ zp, hermitian=True)
    V = ols_vcov(
        zp, resid, bread, kind=vce, clusters=clusters,
        k_total=q + C.shape[1] + int(df_absorbed), nested_adj=nested_adj,
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        effective_n=n_eff, score_scale=score_scale,
    )
    Vinv = np.linalg.pinv(V, hermitian=True)
    wald = float(b.T @ Vinv @ b)
    return {
        "partial_r2": float(r2p),
        "f_classic": float(f_classic),
        "f_classic_pvalue": float(p_classic),
        "f_robust": float(wald / max(df1, 1)),
        "f_robust_chi2_pvalue": float(chi2.sf(wald, df1)),
        "excluded_instruments": int(q),
    }


def _shea_partial_r2(E, C, Z):
    X = np.column_stack([C, E])
    Q = np.column_stack([C, Z])
    xx_inv = np.linalg.pinv(X.T @ X, hermitian=True)
    q_inv = np.linalg.pinv(Q.T @ Q, hermitian=True)
    xpzx = X.T @ Q @ q_inv @ Q.T @ X
    iv_bread = np.linalg.pinv(xpzx, hermitian=True)
    start = C.shape[1]
    num = np.diag(xx_inv)[start:]
    den = np.diag(iv_bread)[start:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.clip(out, 0.0, 1.0)


def first_stage_diagnostics(endog, exog, instruments, *, weights=None, weight_info=None, vce="robust",
                            clusters=None, df_absorbed=0, nested_adj=0, time=None, panel=None,
                            bandwidth=None, kernel="bartlett"):
    E, C, Z = _weighted_triplet(endog, exog, instruments, weights)
    n_eff = float(len(E) if weight_info is None else weight_info.effective_n)
    score_scale = None
    if weight_info is not None and weight_info.kind == "fweight" and str(vce).lower().replace("-", "_") in {"robust", "hc1", "heteroskedastic"}:
        score_scale = 1.0 / np.sqrt(np.asarray(weight_info.estimation, dtype=np.float64))
    shea = _shea_partial_r2(E, C, Z)
    out = []
    for j in range(E.shape[1]):
        d = _first_stage_test(
            E[:, j], C, Z, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
            nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        )
        d["shea_partial_r2"] = float(shea[j])
        out.append(d)
    return out


def sanderson_windmeijer_diagnostics(endog, exog, instruments, *, weights=None, weight_info=None,
                                     vce="robust", clusters=None, df_absorbed=0,
                                     nested_adj=0, time=None, panel=None, bandwidth=None,
                                     kernel="bartlett"):
    """Angrist-Pischke and Sanderson-Windmeijer conditional first-stage tests.

    For one endogenous regressor these reduce to the ordinary excluded-IV test.
    For multiple endogenous regressors the numerator df is L1-K1+1.
    """
    E, C, Z = _weighted_triplet(endog, exog, instruments, weights)
    n, k1 = E.shape
    n_eff = float(n if weight_info is None else weight_info.effective_n)
    score_scale = None
    if weight_info is not None and weight_info.kind == "fweight" and str(vce).lower().replace("-", "_") in {"robust", "hc1", "heteroskedastic"}:
        score_scale = 1.0 / np.sqrt(np.asarray(weight_info.estimation, dtype=np.float64))
    l1 = Z.shape[1]
    df1 = l1 - k1 + 1
    if df1 <= 0:
        return []
    if k1 == 1:
        base = _first_stage_test(
            E[:, 0], C, Z, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
            nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        )
        base["shea_partial_r2"] = float(_shea_partial_r2(E, C, Z)[0])
        return [{"ap": dict(base), "sw": dict(base), "df1": df1}]

    Q = np.column_stack([C, Z])
    coef, *_ = la.lstsq(Q, E, cond=None, lapack_driver="gelsy")
    Ehat = Q @ coef
    X = np.column_stack([C, E])
    Xhat = np.column_stack([C, Ehat])
    out = []
    for j in range(k1):
        target_col = C.shape[1] + j
        keep = np.ones(X.shape[1], dtype=bool)
        keep[target_col] = False
        Xm = X[:, keep]
        Xhm = Xhat[:, keep]
        b1, *_ = la.lstsq(Xhm, E[:, j], cond=None, lapack_driver="gelsy")
        e_ap = E[:, j] - Xhm @ b1
        e_sw = E[:, j] - Xm @ b1
        ap = _first_stage_test(
            e_ap, C, Z, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
            nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            df1_override=df1, effective_n=n_eff, score_scale=score_scale,
        )
        sw = _first_stage_test(
            e_sw, C, Z, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
            nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            df1_override=df1, effective_n=n_eff, score_scale=score_scale,
        )
        out.append({"ap": ap, "sw": sw, "df1": int(df1)})
    return out


def _sym_sqrt(a):
    vals, vecs = np.linalg.eigh((a + a.T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _canonical_components(endog, exog, instruments, weights=None, effective_n=None):
    E, C, Z = _weighted_triplet(endog, exog, instruments, weights)
    n = len(E)
    n_eff = float(n if effective_n is None else effective_n)
    Ep = _residualize_small(E, C)
    Zp = _residualize_small(Z, C)
    qzz = (Zp.T @ Zp) / n_eff
    qzy = (Zp.T @ Ep) / n_eff
    qyy = (Ep.T @ Ep) / n_eff
    try:
        rzz = la.cholesky(qzz, lower=False, check_finite=False)
        ryy = la.cholesky(qyy, lower=False, check_finite=False)
        irzz = la.solve_triangular(rzz, np.eye(rzz.shape[0]), lower=False, check_finite=False)
        iryy = la.solve_triangular(ryy, np.eye(ryy.shape[0]), lower=False, check_finite=False)
    except la.LinAlgError:
        def root_and_invroot(a):
            vals, vecs = np.linalg.eigh((a+a.T)/2)
            tol = np.finfo(float).eps * max(a.shape) * max(float(np.max(np.abs(vals))), 1.0)
            if np.sum(vals > tol) < a.shape[0]:
                raise np.linalg.LinAlgError("rank-deficient canonical-correlation matrix")
            root = (vecs * np.sqrt(vals)) @ vecs.T
            invroot = (vecs / np.sqrt(vals)) @ vecs.T
            return root, invroot
        rzz, irzz = root_and_invroot(qzz)
        ryy, iryy = root_and_invroot(qyy)
    pihat = np.linalg.pinv(qzz, hermitian=True) @ qzy
    theta = rzz @ pihat @ iryy
    return Ep, Zp, pihat, theta, irzz, iryy, C.shape[1], n_eff


def _kron_scores(V, Z, block_rows=100_000):
    n, k1 = V.shape
    l1 = Z.shape[1]
    p = k1 * l1
    out = np.empty((n, p), dtype=np.float64)
    for lo in range(0, n, block_rows):
        hi = min(n, lo + block_rows)
        # Endogenous-major order equals column-major vec(Z'V).
        out[lo:hi] = (V[lo:hi, :, None] * Z[lo:hi, None, :]).reshape(hi-lo, p)
    return out


def _kp_stat(theta, kpvar, n):
    l1, k1 = theta.shape
    U, _, Vh = np.linalg.svd(theta, full_matrices=True)
    V = Vh.T
    kk = k1 - 1
    if kk == 0:
        u12 = np.empty((l1, 0)); u22 = U
        v12 = np.empty((k1, 0)); v22 = V
    else:
        u12 = U[:kk, kk:]
        u22 = U[kk:, kk:]
        v12 = V[:kk, kk:]
        v22 = V[kk:, kk:]
    # Stack the source partitions to reconstruct full row dimension.
    ufull = np.vstack([u12, u22]) if kk else u22
    vfull = np.vstack([v12, v22]) if kk else v22
    us = _sym_sqrt(u22 @ u22.T)
    vs = _sym_sqrt(v22 @ v22.T)
    aq = ufull @ np.linalg.pinv(u22) @ us
    bq = vs @ np.linalg.pinv(v22.T) @ vfull.T
    K = np.kron(bq, aq.T)
    vect = theta.reshape(-1, order="F")
    lam = K @ vect
    vlam = K @ kpvar @ K.T
    rank = int(np.linalg.matrix_rank(vlam))
    stat = float(n * lam.T @ np.linalg.pinv(vlam, hermitian=True) @ lam)
    return stat, rank


def kleibergen_paap_stats(endog, exog, instruments, *, weights=None, weight_info=None, vce="robust",
                           clusters=None, df_absorbed=0, nested_adj=0, time=None, panel=None,
                           bandwidth=None, kernel="bartlett"):
    """Kleibergen-Paap rk LM and rk Wald statistics.

    The covariance of the reduced-form Kronecker scores uses the same
    robust/cluster/HAC/Driscoll-Kraay engine as coefficient inference.
    """
    effective_n = None if weight_info is None else weight_info.effective_n
    Ep, Zp, pihat, theta, irzz, iryy, ccols, n_eff = _canonical_components(
        endog, exog, instruments, weights, effective_n=effective_n
    )
    n, k1 = Ep.shape
    score_scale = None
    if weight_info is not None and weight_info.kind == "fweight" and str(vce).lower().replace("-", "_") in {"robust", "hc1", "heteroskedastic"}:
        score_scale = 1.0 / np.sqrt(np.asarray(weight_info.estimation, dtype=np.float64))
    l1 = Zp.shape[1]
    if l1 < k1:
        return {"rk_lm": np.nan, "rk_lm_df": 0, "rk_lm_pvalue": np.nan,
                "rk_wald_chi2": np.nan, "rk_wald_df": 0, "rk_wald_f": np.nan}
    kind = (vce or "robust").lower().replace("-", "_")
    supported = {
        "robust", "hc1", "heteroskedastic", "cluster",
        "iid", "unadjusted", "homoskedastic",
        "hac", "newey_west", "neweywest",
        "dkraay", "driscoll_kraay", "dk",
    }
    if kind not in supported:
        raise ValueError(f"unsupported KP covariance kind: {vce}")
    if kind in {"dkraay", "driscoll_kraay", "dk"} and time is None:
        raise ValueError("Driscoll-Kraay Kleibergen-Paap statistic requires time=")
    transform = np.kron(iryy.T, irzz.T)

    def stat_for(Vhat):
        scores = _kron_scores(Vhat, Zp)
        if kind in {"iid", "unadjusted", "homoskedastic"}:
            shat = (scores.T @ scores) / n_eff
        else:
            # Stata ivreg2 delegates KP/rank tests to ranktest, whose HAC
            # path uses Bartlett internally even if the coefficient VCE uses
            # another kernel. Preserve that compatibility quirk here.
            kp_kernel = "bartlett" if kind in {
                "hac", "newey_west", "neweywest",
                "dkraay", "driscoll_kraay", "dk"
            } else kernel
            shat = score_covariance(
                scores, kind=kind, clusters=clusters, k_total=0,
                small_sample=False, time=time, panel=panel, bandwidth=bandwidth, kernel=kp_kernel,
                effective_n=n_eff, score_scale=score_scale,
            ) / n_eff
        kpvar = transform @ shat @ transform.T
        return _kp_stat(theta, kpvar, n_eff)

    lm, lm_rank = stat_for(Ep)
    Vrf = Ep - Zp @ pihat
    wald, wald_rank = stat_for(Vrf)
    df = max(l1 - k1 + 1, 1)
    if kind == "cluster" and clusters:
        g = min(len(np.unique(c)) for c in clusters)
        fscale = ((n_eff - (ccols + l1) - int(df_absorbed)) / max(n_eff - 1, 1)) * (max(g - 1, 1) / g)
        wald_f = (wald / max(l1, 1)) * fscale
    else:
        fscale = (n_eff - (ccols + l1) - int(df_absorbed)) / max(n_eff, 1)
        wald_f = (wald / max(l1, 1)) * fscale
    return {
        "rk_lm": float(lm), "rk_lm_df": int(df), "rk_lm_pvalue": float(chi2.sf(lm, df)),
        "rk_wald_chi2": float(wald), "rk_wald_df": int(df), "rk_wald_f": float(wald_f),
        "rank_lm_cov": int(lm_rank), "rank_wald_cov": int(wald_rank),
    }


def cragg_donald_stat(endog, exog, instruments, *, weights=None, df_absorbed=0, effective_n=None):
    E, C, Z = _weighted_triplet(endog, exog, instruments, weights)
    n = E.shape[0]
    n_eff = float(n if effective_n is None else effective_n)
    Ep = _residualize_small(E, C)
    Zp = _residualize_small(Z, C)
    if Zp.shape[1] == 0:
        return np.nan
    zz_inv = np.linalg.pinv(Zp.T @ Zp, hermitian=True)
    epz = Ep.T @ Zp
    explained = epz @ zz_inv @ epz.T
    residual = Ep.T @ Ep - explained
    try:
        eig = la.eigvalsh(explained, residual, check_finite=False)
        cdev = float(np.min(eig))
    except la.LinAlgError:
        cdev = float(np.min(np.linalg.eigvals(np.linalg.pinv(residual) @ explained).real))
    L = C.shape[1] + Zp.shape[1]
    L1 = Zp.shape[1]
    return float(max(n_eff - L - int(df_absorbed), 0) / max(L1, 1) * cdev)


def overid_test(resid, Z, *, X_cols: int, kind="robust", clusters=None, center=False,
                time=None, panel=None, bandwidth=None, kernel="bartlett", effective_n=None, score_scale=None):
    resid = np.asarray(resid, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    df = int(Z.shape[1] - X_cols)
    n_eff = float(len(resid) if effective_n is None else effective_n)
    if df <= 0:
        return {"stat": np.nan, "df": 0, "pvalue": np.nan}
    g = Z.T @ resid
    if kind in (None, "iid", "unadjusted", "homoskedastic"):
        sigma2 = float(resid @ resid) / max(n_eff - X_cols, 1)
        S = sigma2 * (Z.T @ Z)
    else:
        S = score_covariance(
            Z * resid[:, None], kind=kind, clusters=clusters,
            k_total=0, center=center, small_sample=False,
            time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        )
    stat = float(g.T @ np.linalg.pinv(S, hermitian=True) @ g)
    return {"stat": stat, "df": df, "pvalue": float(chi2.sf(stat, df))}
