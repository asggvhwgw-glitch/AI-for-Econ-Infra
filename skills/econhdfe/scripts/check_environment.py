#!/usr/bin/env python3
"""Report the installed econhdfe execution environment as JSON."""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from typing import Any


def module_version(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": str(getattr(mod, "__version__", "unknown"))}


def main() -> int:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "packages": {},
    }
    required = ("econhdfe", "numpy", "scipy", "pandas", "numba", "joblib", "threadpoolctl")
    optional = ("polars", "pyarrow", "cupy")
    for name in required + optional:
        report["packages"][name] = module_version(name)

    try:
        import numba
        report["numba_threads"] = int(numba.get_num_threads())
    except Exception:
        report["numba_threads"] = None

    try:
        import cupy as cp
        report["cuda_device_count"] = int(cp.cuda.runtime.getDeviceCount())
        report["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as exc:
        report["cuda_device_count"] = 0
        report["cuda_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["packages"]["econhdfe"].get("available") else 2


if __name__ == "__main__":
    raise SystemExit(main())
