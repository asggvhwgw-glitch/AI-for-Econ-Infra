from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.planner.calibration import clear_in_memory_calibration_cache, get_thread_calibration


def build_absorber(groups, threads):
    return HDFEAbsorber(
        groups,
        method="map",
        transform="symmetric",
        acceleration="none",
        projection_backend="indexed",
        absorb_threads=threads,
        core_reduction="off",
        max_iter=80,
        tol=1e-8,
    )


def main():
    rng = np.random.default_rng(20260913)
    n = int(os.environ.get("ECONHDFE_BENCH_N", "300000"))
    k = 8
    rounds = int(os.environ.get("ECONHDFE_BENCH_ROUNDS", "5"))
    groups = [
        rng.integers(0, 30_000, size=n, dtype=np.int32),
        rng.integers(0, 120, size=n, dtype=np.int32),
        rng.integers(0, 600, size=n, dtype=np.int32),
        rng.integers(0, 80, size=n, dtype=np.int32),
    ]
    x = rng.normal(size=(n, k))

    cache = Path(os.environ.get("ECONHDFE_CALIBRATION_BENCH_CACHE", "/tmp/econhdfe-thread-calibration-bench.json"))
    cache.unlink(missing_ok=True)
    clear_in_memory_calibration_cache()
    t0 = time.perf_counter()
    calibration = get_thread_calibration(force=True, cache_path=cache)
    cold_seconds = time.perf_counter() - t0
    clear_in_memory_calibration_cache()
    t0 = time.perf_counter()
    cached = get_thread_calibration(cache_path=cache)
    cache_seconds = time.perf_counter() - t0

    labels = [str(t) for t in calibration.candidates] + ["auto"]
    absorbers = {label: build_absorber(groups, "auto" if label == "auto" else int(label)) for label in labels}
    # JIT / allocator warm-up outside timing.
    for absorber in absorbers.values():
        absorber.residualize(x[:, :1], copy=True)

    timings = {label: [] for label in labels}
    order_rng = np.random.default_rng(42)
    for _ in range(rounds):
        order = list(labels)
        order_rng.shuffle(order)
        for label in order:
            t0 = time.perf_counter()
            absorbers[label].residualize(x, copy=True)
            timings[label].append(time.perf_counter() - t0)

    medians = {label: float(np.median(vals)) for label, vals in timings.items()}
    explicit = {label: medians[label] for label in labels if label != "auto"}
    best_explicit = min(explicit.values())
    auto_sec = medians["auto"]
    auto_resolved = int(absorbers["auto"].absorb_threads)

    payload = {
        "nobs": n,
        "ncols": k,
        "fe_dimensions": len(groups),
        "rounds": rounds,
        "calibration": calibration.as_dict(),
        "cache_reload_source": cached.source,
        "calibration_cold_seconds": cold_seconds,
        "calibration_cache_reload_seconds": cache_seconds,
        "hdfe_median_seconds_by_explicit_threads": explicit,
        "hdfe_all_round_seconds": timings,
        "hdfe_resolved_threads": {label: int(absorbers[label].absorb_threads) for label in labels},
        "auto_resolved_threads": auto_resolved,
        "auto_hdfe_median_seconds": auto_sec,
        "best_explicit_hdfe_seconds": best_explicit,
        "auto_to_best_explicit_ratio": auto_sec / best_explicit,
        "auto_near_best_10pct": bool(auto_sec <= best_explicit * 1.10),
    }
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
