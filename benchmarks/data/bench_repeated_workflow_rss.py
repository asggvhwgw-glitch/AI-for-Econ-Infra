from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def make_data(path: Path, *, n=100_000, raw_columns=64, seed=19021):
    rng = np.random.default_rng(seed)
    firm = np.arange(n) % 10_000
    year = (np.arange(n) // 10_000) % 10
    data = {"firm": firm.astype(str), "year": year}
    for j in range(8):
        data[f"x{j}"] = rng.normal(size=n)
    common = rng.normal(size=n)
    for j in range(5):
        data[f"y{j}"] = 0.3 * data[f"x{j % 8}"] + 0.1 * common + rng.normal(size=n)
    for j in range(raw_columns - len(data)):
        data[f"unused{j}"] = rng.normal(size=n)
    pd.DataFrame(data).to_csv(path, index=False)


def specs(count=30):
    out = []
    for r in range(count):
        out.append({
            "y": f"y{r % 5}",
            "x": [f"x{j}" for j in range(2 + r % 6)],
            "absorb": ["firm", "year"],
        })
    return out


def worker(path: Path, mode: str):
    from econhdfe import OLSHDFESession
    from econhdfe.config import ExecutionConfig
    from econhdfe.data import CSVSource

    cfg = ExecutionConfig(cache_validation="signature", threads=1, memory_budget_mb=512)
    table = specs()
    t0 = time.perf_counter()
    if mode == "baseline":
        data = pd.read_csv(path)
        sess = OLSHDFESession(data, execution_config=cfg, drop_singletons=False)
    else:
        sess = OLSHDFESession(CSVSource(path), execution_config=cfg, drop_singletons=False)
    sess.fit_many(table)
    elapsed = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"mode": mode, "seconds": elapsed, "maxrss_kb": int(rss), "cache": sess.cache_info()}


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print(json.dumps(worker(Path(sys.argv[2]), sys.argv[3]), default=str))
        return

    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "wide.csv"
        make_data(path)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        results = {}
        for mode in ("baseline", "encoded"):
            cp = subprocess.run(
                [sys.executable, __file__, "--worker", str(path), mode],
                check=True, capture_output=True, text=True, env=env,
            )
            results[mode] = json.loads(cp.stdout)
        b, e = results["baseline"], results["encoded"]
        out = {
            "nobs": 100_000,
            "raw_columns": 64,
            "specifications": 30,
            "baseline": b,
            "encoded": e,
            "speedup": b["seconds"] / e["seconds"],
            "peak_rss_reduction_fraction": 1.0 - e["maxrss_kb"] / b["maxrss_kb"],
        }
        target = root / "benchmarks" / "data" / "repeated_workflow_rss.json"
        target.write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
