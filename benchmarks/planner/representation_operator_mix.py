from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock


def _make_design(n: int, blocks: int, local: int, shared: int, seed: int):
    rng = np.random.default_rng(seed)
    row_parts = np.array_split(np.arange(n, dtype=np.int64), blocks)
    k = blocks * local + shared
    dense = np.zeros((n, k), dtype=np.float64)
    out = []
    for b, rows in enumerate(row_parts):
        cols = np.r_[np.arange(b * local, (b + 1) * local), np.arange(blocks * local, k)].astype(np.int64)
        values = rng.normal(size=(len(rows), len(cols)))
        dense[np.ix_(rows, cols)] = values
        out.append(DenseDesignBlock(b, rows, cols, values))
    return dense, BlockDesign._from_trusted(n, k, tuple(out))


def _best(fn, reps: int = 5) -> float:
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(min(vals))


def bench_case(*, n: int, blocks: int, local: int, shared: int, blas_threads: int, seed: int):
    X, B = _make_design(n, blocks, local, shared, seed)
    beta = np.ones(X.shape[1])
    y = np.ones(n)
    with threadpool_limits(limits=blas_threads):
        X @ beta; B.matvec(beta); X.T @ y; B.t_matvec(y)
        dense_mv = _best(lambda: X @ beta)
        block_mv = _best(lambda: B.matvec(beta))
        dense_tmv = _best(lambda: X.T @ y)
        block_tmv = _best(lambda: B.t_matvec(y))
    return {
        "nobs": n,
        "blocks": blocks,
        "local_width": local,
        "shared_width": shared,
        "ncols": int(X.shape[1]),
        "blas_threads": int(blas_threads),
        "dense_bytes": int(X.nbytes),
        "block_bytes": int(B.nbytes),
        "storage_savings_fraction": float(1.0 - B.nbytes / X.nbytes),
        "dense_matvec_seconds": dense_mv,
        "block_matvec_seconds": block_mv,
        "dense_over_block_matvec": float(dense_mv / block_mv),
        "dense_tmatvec_seconds": dense_tmv,
        "block_tmatvec_seconds": block_tmv,
        "dense_over_block_tmatvec": float(dense_tmv / block_tmv),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    ap.add_argument("--nobs", type=int, default=120_000)
    args = ap.parse_args()
    cases = [(4, 8, 0), (6, 8, 0), (6, 8, 4), (12, 8, 2), (12, 8, 8)]
    rows = []
    for threads in (1, 4):
        for i, (blocks, local, shared) in enumerate(cases):
            rows.append(bench_case(
                n=args.nobs, blocks=blocks, local=local, shared=shared,
                blas_threads=threads, seed=1700 + i,
            ))
    payload = {
        "purpose": "calibrate whether byte-traffic alone predicts dense-vs-block operator performance",
        "interpretation": "development microbenchmark; not a release performance promise",
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
