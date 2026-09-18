from __future__ import annotations
import math
import numpy as np


_LAG_KERNELS = {
    "bartlett", "parzen", "truncated", "tukey_hanning", "tukey_hamming"
}
_SPECTRAL_KERNELS = {"quadratic_spectral", "daniell", "tent"}


def automatic_bandwidth(n_periods: int) -> int:
    """reghdfe-style default HAC/DK bandwidth (bandwidth = max lag + 1)."""
    t = max(int(n_periods), 1)
    return max(1, int(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0))) + 1)


def _canonical_kernel(kernel: str) -> str:
    name = str(kernel).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bar": "bartlett", "bartlett": "bartlett", "newey_west": "bartlett", "neweywest": "bartlett", "nw": "bartlett",
        "par": "parzen", "parzen": "parzen",
        "tru": "truncated", "truncated": "truncated", "uniform": "truncated", "rectangular": "truncated",
        "thann": "tukey_hanning", "tukey_hanning": "tukey_hanning",
        "thamm": "tukey_hamming", "tukey_hamming": "tukey_hamming",
        "qua": "quadratic_spectral", "qs": "quadratic_spectral", "quadratic_spectral": "quadratic_spectral",
        "dan": "daniell", "daniell": "daniell", "danielle": "daniell",
        "ten": "tent", "tent": "tent",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        valid = "bartlett, truncated, parzen, tukey-hanning, tukey-hamming, quadratic-spectral, daniell, tent"
        raise ValueError(f"kernel must be one of: {valid}") from exc


def kernel_type(kernel: str) -> str:
    return "spectral" if _canonical_kernel(kernel) in _SPECTRAL_KERNELS else "lag"


def kernel_weight(lag: int, bandwidth: float, kernel: str = "bartlett") -> float:
    """ivreg2-compatible HAC/AC kernel weight for a non-negative integer lag.

    Lag-window kernels have compact support. Quadratic-Spectral, Daniell and
    Tent are spectral-window kernels and retain nonzero weights beyond the
    nominal bandwidth, so callers must not truncate them at ``bandwidth-1``.
    """
    lag = abs(int(lag))
    if lag == 0:
        return 1.0
    b = float(bandwidth)
    if not np.isfinite(b) or b <= 0:
        raise ValueError("bandwidth must be positive")
    x = float(lag) / b
    name = _canonical_kernel(kernel)

    if name == "bartlett":
        return max(1.0 - x, 0.0)
    if name == "truncated":
        return 1.0 if x <= 1.0 else 0.0
    if name == "parzen":
        if x <= 0.5:
            return 1.0 - 6.0 * x * x + 6.0 * x * x * x
        if x <= 1.0:
            return 2.0 * (1.0 - x) ** 3
        return 0.0
    if name == "tukey_hanning":
        return 0.5 + 0.5 * math.cos(math.pi * x) if x <= 1.0 else 0.0
    if name == "tukey_hamming":
        return 0.54 + 0.46 * math.cos(math.pi * x) if x <= 1.0 else 0.0
    if name == "quadratic_spectral":
        # Stable form of 25/(12*pi^2*x^2) * (sin(y)/y - cos(y)), y=6*pi*x/5.
        if abs(x) < 1e-3:
            x0 = 1e-3
            y0 = 6.0 * math.pi * x0 / 5.0
            q0 = (25.0 / (12.0 * math.pi**2 * x0**2)) * (math.sin(y0) / y0 - math.cos(y0))
            return math.exp(1e6 * math.log(q0) * x * x)
        y = 6.0 * math.pi * x / 5.0
        return (25.0 / (12.0 * math.pi**2 * x * x)) * (math.sin(y) / y - math.cos(y))
    if name == "daniell":
        z = math.pi * x
        return 1.0 if abs(z) < 1e-10 else math.sin(z) / z
    # Tent: reproduce ivreg2's m_calckw convention where lag appears both
    # explicitly and via karg=lag/bandwidth.
    karg = x
    if abs(karg) < 1e-10:
        return float(lag * lag)
    return 2.0 * (1.0 - math.cos(lag * karg)) / (karg * karg)


def _numeric_time_values(unique_time: np.ndarray):
    a = np.asarray(unique_time)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[ns]").astype(np.int64)
    if np.issubdtype(a.dtype, np.timedelta64):
        return a.astype("timedelta64[ns]").astype(np.int64)
    if np.issubdtype(a.dtype, np.number):
        return a.astype(np.float64, copy=False)
    return None


def _time_step_and_span(unique_time: np.ndarray) -> tuple[float | None, float | None]:
    vals = _numeric_time_values(unique_time)
    if vals is None or len(vals) <= 1:
        return None, None
    diffs = np.diff(vals)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return None, None
    delta = float(np.min(positive))
    span = float(vals[-1] - vals[0])
    return delta, span


def _max_lag(n_periods: int, bandwidth: float, kernel: str, unique_time=None) -> int:
    if n_periods <= 1:
        return 0
    if kernel_type(kernel) == "lag":
        # ivreg2 iterates floor(bw); compact kernels may assign the boundary
        # zero weight (Bartlett/Parzen/Tukey) while Truncated keeps it.
        return min(n_periods - 1, max(int(math.floor(float(bandwidth))), 0))
    if unique_time is not None:
        delta, span = _time_step_and_span(np.asarray(unique_time))
        if delta is not None and delta > 0:
            return min(n_periods - 1, max(int(span / delta), 0))
    return n_periods - 1


def _time_cross(time_scores: np.ndarray, unique_time: np.ndarray, lag: int) -> np.ndarray:
    """Cross product S_t' S_{t-lag}, preserving gaps for numeric time indexes."""
    if lag <= 0:
        return time_scores.T @ time_scores
    vals = _numeric_time_values(unique_time)
    if vals is None or len(vals) <= 1:
        if lag >= len(time_scores):
            return np.zeros((time_scores.shape[1], time_scores.shape[1]))
        return time_scores[lag:].T @ time_scores[:-lag]
    diffs = np.diff(vals)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return np.zeros((time_scores.shape[1], time_scores.shape[1]))
    delta = np.min(positive)
    target = vals - lag * delta
    pos = np.searchsorted(vals, target)
    valid = pos < len(vals)
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        valid[valid_idx] &= vals[pos[valid_idx]] == target[valid_idx]
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        return np.zeros((time_scores.shape[1], time_scores.shape[1]))
    return time_scores[rows].T @ time_scores[pos[rows]]


def _prepare_hac(scores: np.ndarray, time=None, panel=None):
    scores = np.asarray(scores, dtype=np.float64)
    if time is None:
        if panel is not None:
            raise ValueError("panel= requires time= for HAC covariance")
        return scores, None, None
    t = np.asarray(time)
    if len(t) != len(scores):
        raise ValueError("time must have the same number of observations as scores")
    if panel is None:
        order = np.argsort(t, kind="stable")
        t = t[order]
        scores = scores[order]
        if len(t) > 1 and np.any(t[1:] == t[:-1]):
            raise ValueError("HAC time contains duplicates; provide panel= for panel time series")
        return scores, t, None
    p_raw = np.asarray(panel)
    if len(p_raw) != len(scores):
        raise ValueError("panel must have the same number of observations as scores")
    _, pcode = np.unique(p_raw, return_inverse=True)
    order = np.lexsort((t, pcode))
    t = t[order]
    pcode = pcode[order].astype(np.int64, copy=False)
    scores = scores[order]
    if len(t) > 1:
        duplicate = (pcode[1:] == pcode[:-1]) & (t[1:] == t[:-1])
        if np.any(duplicate):
            raise ValueError("duplicate (panel, time) combinations are not allowed for HAC")
    return scores, t, pcode


def _panel_time_cross(scores: np.ndarray, time: np.ndarray, panel: np.ndarray, lag: int) -> np.ndarray:
    vals = _numeric_time_values(time)
    if vals is None:
        raise ValueError("panel-aware HAC requires numeric, datetime64, or timedelta64 time")
    if len(vals) <= 1:
        return np.zeros((scores.shape[1], scores.shape[1]))
    same = panel[1:] == panel[:-1]
    diffs = np.diff(vals)[same]
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return np.zeros((scores.shape[1], scores.shape[1]))
    delta = np.min(positive)
    # Structured search gives vectorized exact matching of (panel, t-lag*delta)
    # and prevents accidental cross-panel pairs at panel boundaries.
    dtype = np.dtype([("p", np.int64), ("t", vals.dtype)])
    keys = np.empty(len(vals), dtype=dtype)
    keys["p"], keys["t"] = panel, vals
    target = np.empty(len(vals), dtype=dtype)
    target["p"], target["t"] = panel, vals - lag * delta
    pos = np.searchsorted(keys, target)
    valid = pos < len(keys)
    idx = np.flatnonzero(valid)
    if idx.size:
        valid[idx] &= keys[pos[idx]] == target[idx]
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        return np.zeros((scores.shape[1], scores.shape[1]))
    return scores[rows].T @ scores[pos[rows]]


def hac_meat(scores: np.ndarray, *, bandwidth=None, kernel="bartlett", time=None, panel=None) -> tuple[np.ndarray, float]:
    """HAC meat without constructing lagged N x N matrices.

    With ``panel=None`` and an explicit ``time``, time values must be unique.
    With ``panel`` supplied, lag pairs are matched within panels using exact
    ``(panel, time-lag)`` keys, so unbalanced/gapped panels do not create
    cross-panel or false adjacent-row autocovariances.
    """
    scores, times, panels = _prepare_hac(scores, time=time, panel=panel)
    n = len(scores)
    n_periods = n if times is None else len(np.unique(times))
    b = float(automatic_bandwidth(n_periods) if bandwidth is None else bandwidth)
    max_lag = _max_lag(n_periods, b, kernel, np.unique(times) if times is not None else None)
    meat = scores.T @ scores
    for lag in range(1, max_lag + 1):
        kw = kernel_weight(lag, b, kernel)
        if kw == 0.0:
            continue
        if times is None:
            cross = scores[lag:].T @ scores[:-lag]
        elif panels is None:
            cross = _time_cross(scores, times, lag)
        else:
            cross = _panel_time_cross(scores, times, panels, lag)
        meat += kw * (cross + cross.T)
    return (meat + meat.T) / 2.0, b

def driscoll_kraay_meat(scores: np.ndarray, time, *, bandwidth=None, kernel="bartlett") -> tuple[np.ndarray, float, int]:
    """Driscoll-Kraay meat from cross-sectional sums of moment scores.

    The implementation is O(NK + B*T*K^2) for compact kernels. Spectral
    kernels use the full available time span, matching ivreg2's spectral
    window convention, still without any N x N allocation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    time = np.asarray(time)
    if len(time) != len(scores):
        raise ValueError("time must have the same number of observations as scores")
    unique, codes = np.unique(time, return_inverse=True)
    t = len(unique)
    sums = np.empty((t, scores.shape[1]), dtype=np.float64)
    for j in range(scores.shape[1]):
        sums[:, j] = np.bincount(codes, weights=scores[:, j], minlength=t)
    b = float(automatic_bandwidth(t) if bandwidth is None else bandwidth)
    max_lag = _max_lag(t, b, kernel, unique)
    meat = sums.T @ sums
    for lag in range(1, max_lag + 1):
        kw = kernel_weight(lag, b, kernel)
        if kw == 0.0:
            continue
        cross = _time_cross(sums, unique, lag)
        meat += kw * (cross + cross.T)
    return (meat + meat.T) / 2.0, b, t
