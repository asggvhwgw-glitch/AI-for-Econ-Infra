from __future__ import annotations
import json
from time import perf_counter
import numpy as np
import pandas as pd
from econhdfe.data import DataFrameSource, materialize_required_data


def main():
    n = 500_000
    firm = np.array([f"firm_{i % 50_000:05d}" for i in range(n)], dtype=object)
    year = np.array([f"year_{2000 + (i % 20)}" for i in range(n)], dtype=object)
    df = pd.DataFrame({"firm": firm, "year": year, "x": np.arange(n, dtype=np.float64)})
    raw_bytes = int(df[["firm", "year", "x"]].memory_usage(index=False, deep=True).sum())
    t0 = perf_counter()
    out = materialize_required_data(
        DataFrameSource(df), ["firm", "year", "x"], memory_budget_mb=128,
        identifier_columns=["firm", "year"],
    )
    elapsed = perf_counter() - t0
    encoded_bytes = int(out.materialized_bytes)
    result = {
        "nobs": n,
        "firm_levels": len(out.identifier_levels["firm"]),
        "year_levels": len(out.identifier_levels["year"]),
        "raw_payload_mb": raw_bytes / 1024**2,
        "encoded_payload_mb": encoded_bytes / 1024**2,
        "payload_reduction": 1.0 - encoded_bytes / raw_bytes,
        "encoding_seconds": elapsed,
    }
    print(json.dumps(result, indent=2))
    from pathlib import Path
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
