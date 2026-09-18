from __future__ import annotations
from collections.abc import Callable, Sequence
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from ..errors import EconHDFEError, error_boundary
from .results import ReplicateBatch, ReplicateFailure


def _failure(exc: BaseException) -> ReplicateFailure:
    if isinstance(exc, EconHDFEError):
        code = exc.code
    elif isinstance(exc, np.linalg.LinAlgError):
        code = "numerical.linalg"
    elif isinstance(exc, FloatingPointError):
        code = "numerical.floating_point"
    elif isinstance(exc, OverflowError):
        code = "numerical.overflow"
    else:
        code = f"resampling.{type(exc).__name__.lower()}"
    return ReplicateFailure(code=code, type=type(exc).__name__, message=str(exc))


def run_replicates(
    worker: Callable[[np.random.SeedSequence], object],
    seeds: Sequence[np.random.SeedSequence], *, n_jobs: int = 1,
    skip_exceptions=(EconHDFEError, np.linalg.LinAlgError, FloatingPointError, OverflowError),
) -> ReplicateBatch:
    """Run independent replicates without hiding programming errors.

    Only explicitly declared statistical/numerical failures are skippable.
    AttributeError, NameError, AssertionError, etc. propagate immediately.
    """
    def one(ss):
        try:
            return True, worker(ss)
        except skip_exceptions as exc:
            return False, _failure(exc)

    if n_jobs == 1:
        raw = [one(s) for s in seeds]
    else:
        with threadpool_limits(limits=1):
            raw = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(s) for s in seeds)
    batch = ReplicateBatch()
    for ok, value in raw:
        if ok:
            batch.values.append(value)
        else:
            batch.failures.append(value)
    return batch


@error_boundary("resampling")
def parallel_pairs_bootstrap(estimator, nobs: int, *, reps=999, n_jobs=-1, seed=0):
    seeds = np.random.SeedSequence(seed).spawn(reps)
    def one(ss):
        rng = np.random.default_rng(ss)
        idx = rng.integers(0, nobs, size=nobs)
        with threadpool_limits(limits=1):
            return np.asarray(estimator(idx), dtype=np.float64)
    batch = run_replicates(one, seeds, n_jobs=n_jobs, skip_exceptions=())
    return np.vstack(batch.values)
