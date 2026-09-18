from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.compute.partitioned_lstsq import partitioned_weighted_lstsq, plan_partitioned_wls


def run(*, blocks=8, per_block=1200, local_width=4, shared_widths=(1, 2, 4, 8, 16, 32, 64), reps=5):
    out = []
    for shared in shared_widths:
        rng = np.random.default_rng(6000 + int(shared))
        ncols = blocks * local_width + int(shared)
        parts = []
        for b in range(blocks):
            rows = np.arange(b * per_block, (b + 1) * per_block)
            cols = np.r_[
                np.arange(b * local_width, (b + 1) * local_width),
                np.arange(blocks * local_width, ncols),
            ]
            parts.append(DenseDesignBlock(b, rows, cols, rng.normal(size=(per_block, len(cols)))))
        design = BlockDesign(blocks * per_block, ncols, tuple(parts))
        X = design.materialize()
        y = rng.normal(size=design.nobs)
        w = np.exp(rng.normal(scale=.2, size=design.nobs))
        sw = np.sqrt(w)

        partitioned_weighted_lstsq(design, y, w, chunk_rows=500)
        np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        tb = []
        td = []
        for _ in range(reps):
            t0 = time.perf_counter()
            partitioned_weighted_lstsq(design, y, w, chunk_rows=500)
            tb.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            td.append(time.perf_counter() - t0)
        plan = plan_partitioned_wls(design)
        out.append({
            "shared_width": int(shared),
            "ncols": int(ncols),
            "block_median_seconds": float(np.median(tb)),
            "dense_median_seconds": float(np.median(td)),
            "dense_over_block_speedup": float(np.median(td) / np.median(tb)),
            "payload_mib": design.payload_bytes / (1024**2),
            "dense_mib": design.dense_equivalent_bytes / (1024**2),
            "solve_plan": plan.as_dict(),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    result = run(reps=args.reps)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
