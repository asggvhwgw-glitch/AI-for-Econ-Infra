from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import json, resource, sys
from time import perf_counter
import pandas as pd
path = sys.argv[1]
mode = sys.argv[2]
wanted = tuple(sys.argv[3].split(','))
if mode != 'full':
    from econhdfe.data import CSVSource, materialize_required_data
t0 = perf_counter()
if mode == 'full':
    frame = pd.read_csv(path)
    extra = {}
else:
    out = materialize_required_data(CSVSource(path), wanted, memory_budget_mb=128)
    frame = out.frame
    extra = {
        'planned_batch_rows': out.plan.batch_rows,
        'required_column_count': out.plan.projected_column_count,
    }
elapsed = perf_counter() - t0
print(json.dumps({
    'seconds': elapsed,
    'dataframe_mb': int(frame.memory_usage(index=False, deep=True).sum()) / 1024**2,
    'peak_rss_mb': float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
    **extra,
}))
'''


def child(path: Path, mode: str, wanted: tuple[str, ...]):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    proc = subprocess.run(
        [sys.executable, '-c', CHILD, str(path), mode, ','.join(wanted)],
        check=True, capture_output=True, text=True, env=env,
    )
    return json.loads(proc.stdout)


def main():
    rng = np.random.default_rng(260913)
    n = 120_000
    k = 64
    wanted = ("x0", "x1", "x2", "x3", "x4", "x5")
    with TemporaryDirectory() as td:
        path = Path(td) / "wide.csv"
        arr = rng.normal(size=(n, k))
        pd.DataFrame(arr, columns=[f"x{j}" for j in range(k)]).to_csv(path, index=False)
        del arr
        full = child(path, 'full', wanted)
        projected = child(path, 'projected', wanted)
        out = {
            "nobs": n,
            "source_columns": k,
            "required_columns": len(wanted),
            "csv_size_mb": path.stat().st_size / 1024**2,
            "full": full,
            "projected": projected,
            "speed_ratio_full_over_projected": full["seconds"] / projected["seconds"],
            "payload_reduction": 1.0 - projected["dataframe_mb"] / full["dataframe_mb"],
            "peak_rss_reduction": 1.0 - projected["peak_rss_mb"] / full["peak_rss_mb"],
        }
        print(json.dumps(out, indent=2))
        Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
