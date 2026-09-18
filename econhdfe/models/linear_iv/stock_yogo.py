from __future__ import annotations
import math

# Stock-Yogo (2005) / ivreg2 compatibility values for the most common designs.
# Rows are L1=1..10 excluded instruments.  This compact v0.3 table intentionally
# covers common applications while keeping provenance explicit; the validation
# framework can compare against full Stata tables when available.
_N = float("nan")

_IV_SIZE = {
    10: [(16.38,_N),(19.93,7.03),(22.30,13.43),(24.58,16.87),(26.87,19.45),
         (29.18,21.68),(31.50,23.72),(33.84,25.64),(36.19,27.51),(38.54,29.32)],
    15: [(8.96,_N),(11.59,4.58),(12.83,8.18),(13.96,9.93),(15.09,11.22),
         (16.23,12.33),(17.38,13.34),(18.54,14.31),(19.71,15.24),(20.88,16.16)],
    20: [(6.66,_N),(8.75,3.95),(9.54,6.40),(10.26,7.54),(10.98,8.38),
         (11.72,9.10),(12.48,9.77),(13.24,10.41),(14.01,11.03),(14.78,11.65)],
    25: [(5.53,_N),(7.25,3.63),(7.80,5.45),(8.31,6.28),(8.84,6.89),
         (9.38,7.42),(9.93,7.91),(10.50,8.39),(11.07,8.85),(11.65,9.31)],
}

_IV_BIAS = {
    5: [(_N,_N,_N),(_N,_N,_N),(13.91,_N,_N),(16.85,11.04,_N),(18.37,13.97,9.53),
        (19.28,15.72,12.20),(19.86,16.88,13.95),(20.25,17.70,15.18),(20.53,18.30,16.10),(20.74,18.76,16.80)],
    10:[(_N,_N,_N),(_N,_N,_N),(9.08,_N,_N),(10.27,7.56,_N),(10.83,8.78,6.61),
        (11.12,9.48,7.77),(11.29,9.92,8.50),(11.39,10.22,9.01),(11.46,10.43,9.37),(11.49,10.58,9.64)],
    20:[(_N,_N,_N),(_N,_N,_N),(6.46,_N,_N),(6.71,5.57,_N),(6.77,5.91,4.99),
        (6.76,6.08,5.35),(6.73,6.16,5.56),(6.69,6.20,5.69),(6.65,6.22,5.78),(6.61,6.23,5.83)],
    30:[(_N,_N,_N),(_N,_N,_N),(5.39,_N,_N),(5.34,4.73,_N),(5.25,4.79,4.30),
        (5.15,4.78,4.40),(5.07,4.76,4.44),(4.99,4.73,4.46),(4.92,4.69,4.46),(4.86,4.66,4.45)],
}


def stock_yogo_critical_values(l1: int, k1: int, *, estimator="2sls") -> dict:
    """Return bundled Stock-Yogo weak-ID critical values.

    v0.3 bundles the common 2SLS table subset L1<=10 (size K1<=2, relative
    bias K1<=3). Values are intended for the homoskedastic Cragg-Donald F.
    Applying them to KP statistics under robust/cluster/HAC errors is only a
    heuristic, as in ivreg2's cautionary documentation.
    """
    l1, k1 = int(l1), int(k1)
    est = estimator.lower()
    if est not in {"2sls", "iv", "gmm2s"}:
        return {"available": False, "reason": "bundled v0.3 subset is for 2SLS/GMM2S only"}
    if not (1 <= l1 <= 10):
        return {"available": False, "reason": "bundled v0.3 Stock-Yogo subset covers L1=1..10"}
    size = {}
    if 1 <= k1 <= 2:
        for threshold, rows in _IV_SIZE.items():
            v = rows[l1-1][k1-1]
            if not math.isnan(v):
                size[f"{threshold}%"] = float(v)
    bias = {}
    if 1 <= k1 <= 3:
        for threshold, rows in _IV_BIAS.items():
            v = rows[l1-1][k1-1]
            if not math.isnan(v):
                bias[f"{threshold}%"] = float(v)
    return {
        "available": bool(size or bias),
        "l1": l1,
        "k1": k1,
        "maximal_iv_size": size,
        "maximal_relative_bias": bias,
        "applicability": "Cragg-Donald F under i.i.d./homoskedastic errors; robust/KP comparison is heuristic",
        "coverage": "v0.3 bundled common subset: L1<=10",
    }
