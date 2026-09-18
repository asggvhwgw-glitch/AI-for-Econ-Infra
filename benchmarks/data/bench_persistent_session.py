from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd


def _run(repo: Path, code: str, env: dict) -> dict:
    t0 = time.perf_counter()
    cp = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, env=env,
        text=True, capture_output=True, check=True,
    )
    wall = time.perf_counter() - t0
    payload = json.loads(cp.stdout.strip().splitlines()[-1])
    payload["process_wall_seconds"] = wall
    return payload


def main(n=220_000, raw_columns=48, seed=93013, source_validation="strict"):
    repo = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(seed)
    firm = np.arange(n) % 22_000
    year = (np.arange(n) // 22_000) % 10
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    a = rng.normal(size=22_000)[firm]; d = rng.normal(size=10)[year]
    data = {
        "firm": np.asarray([f"f{v}" for v in firm], dtype=object),
        "year": year, "x1": x1, "x2": x2,
        "y1": 0.55*x1 + a + d + rng.normal(scale=.6, size=n),
        "y2": -0.25*x1 + 0.15*x2 + a + d + rng.normal(scale=.6, size=n),
    }
    for j in range(raw_columns-len(data)):
        data[f"unused{j}"] = rng.normal(size=n)
    frame = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td); csv = td / "research.csv"; cache = td / "cache"
        frame.to_csv(csv, index=False)
        env = os.environ.copy(); env["PYTHONPATH"] = str(repo)

        common = f'''\nimport json, time\nfrom econhdfe import OLSHDFESession\nfrom econhdfe.data import CSVSource\nfrom econhdfe.config import ExecutionConfig\npath=r"{csv}"\ncache=r"{cache}"\ncfg=ExecutionConfig(cache_validation="signature",threads=1,memory_budget_mb=512)\nvalidation="{source_validation}"\n'''
        baseline_change_y_code = common + '''\nt=time.perf_counter()\ns=OLSHDFESession(CSVSource(path),execution_config=cfg,drop_singletons=False)\nr=s.fit(y="y2",x=["x1"],absorb=["firm","year"])\nprint(json.dumps({"inner_seconds":time.perf_counter()-t,"params":r.params.tolist(),"cache":s.cache_info()},default=str))\n'''
        baseline_add_control_code = common + '''\nt=time.perf_counter()\ns=OLSHDFESession(CSVSource(path),execution_config=cfg,drop_singletons=False)\nr=s.fit(y="y2",x=["x1","x2"],absorb=["firm","year"])\nprint(json.dumps({"inner_seconds":time.perf_counter()-t,"params":r.params.tolist(),"cache":s.cache_info()},default=str))\n'''
        build_code = common + '''\nt=time.perf_counter()\ns=OLSHDFESession(CSVSource(path),execution_config=cfg,drop_singletons=False).enable_persistent_cache(cache,source_validation=validation)\nr=s.fit(y="y1",x=["x1"],absorb=["firm","year"])\nprint(json.dumps({"inner_seconds":time.perf_counter()-t,"params":r.params.tolist(),"persistent":s.persistent_cache_info(),"cache":s.cache_info()},default=str))\n'''
        change_y_code = common + '''\nt=time.perf_counter()\ns=OLSHDFESession(CSVSource(path),execution_config=cfg,drop_singletons=False).enable_persistent_cache(cache,source_validation=validation)\nr=s.fit(y="y2",x=["x1"],absorb=["firm","year"])\nprint(json.dumps({"inner_seconds":time.perf_counter()-t,"params":r.params.tolist(),"persistent":s.persistent_cache_info(),"cache":s.cache_info()},default=str))\n'''
        add_control_code = common + '''\nt=time.perf_counter()\ns=OLSHDFESession(CSVSource(path),execution_config=cfg,drop_singletons=False).enable_persistent_cache(cache,source_validation=validation)\nr=s.fit(y="y2",x=["x1","x2"],absorb=["firm","year"])\nprint(json.dumps({"inner_seconds":time.perf_counter()-t,"params":r.params.tolist(),"persistent":s.persistent_cache_info(),"cache":s.cache_info()},default=str))\n'''
        exact_resume_code = add_control_code

        baseline_change_y = _run(repo, baseline_change_y_code, env)
        baseline_add_control = _run(repo, baseline_add_control_code, env)
        build = _run(repo, build_code, env)
        change_y = _run(repo, change_y_code, env)
        add_control = _run(repo, add_control_code, env)
        exact_resume = _run(repo, exact_resume_code, env)

        cache_bytes = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
        out = {
            "nobs": n, "raw_columns": raw_columns, "source_validation": source_validation,
            "cache_bytes_after_workflow": cache_bytes,
            "baseline_change_y_new_process": baseline_change_y,
            "baseline_add_control_new_process": baseline_add_control,
            "persistent_initial_build": build,
            "persistent_change_y_same_x": change_y,
            "persistent_add_control": add_control,
            "persistent_exact_resume": exact_resume,
            "change_y_speedup_vs_baseline_inner": baseline_change_y["inner_seconds"] / change_y["inner_seconds"],
            "add_control_speedup_vs_baseline_inner": baseline_add_control["inner_seconds"] / add_control["inner_seconds"],
            "exact_resume_speedup_vs_baseline_inner": baseline_add_control["inner_seconds"] / exact_resume["inner_seconds"],
            "change_y_speedup_vs_baseline_process": baseline_change_y["process_wall_seconds"] / change_y["process_wall_seconds"],
            "add_control_speedup_vs_baseline_process": baseline_add_control["process_wall_seconds"] / add_control["process_wall_seconds"],
            "exact_resume_speedup_vs_baseline_process": baseline_add_control["process_wall_seconds"] / exact_resume["process_wall_seconds"],
        }
        dest = repo / f"benchmarks/data/persistent_session_resume_{source_validation}.json"
        dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
