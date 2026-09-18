"""Installed-wheel smoke test for privacy-safe support reporting."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from econhdfe import ExecutionConfig, __version__
from econhdfe.errors import NumericalError
from econhdfe.results import RegressionResult
from econhdfe.support_reports import (
    benchmark_report_template,
    build_error_report,
    error_report_template,
    write_template,
)


def main() -> None:
    result = RegressionResult(
        params=np.array([1234.5]), vcov=np.eye(1), stderr=np.ones(1),
        residuals=np.array([9876.5]), fitted=np.array([555.0]),
        nobs=10, rank=1, df_resid=8.0, df_absorbed=1,
        converged=True, iterations=3, dropped_singletons=0,
        names=("secret_x",), fe_names=("secret_fe",),
    )
    error = NumericalError(
        "secret /private/path customer_id",
        details={"iterations": 4, "account_number": 123456789},
    )
    report = build_error_report(
        error=error, result=result,
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=128),
    )
    for forbidden in (
        "secret_x", "secret_fe", "/private/path", "customer_id",
        "123456789", "1234.5", "9876.5",
    ):
        assert forbidden not in report, forbidden
    assert "econhdfe.parameter_error.v1" in report
    assert "iterations" in report
    assert error_report_template().startswith("# econhdfe privacy-minimized")
    assert benchmark_report_template().startswith("# econhdfe real-machine benchmark report")
    with tempfile.TemporaryDirectory(prefix="econhdfe-report-smoke-") as td:
        e = Path(td) / "error.md"
        b = Path(td) / "benchmark.md"
        write_template("error", e)
        write_template("benchmark", b)
        assert e.read_text() == error_report_template()
        assert b.read_text() == benchmark_report_template()
    print(f"support-report smoke PASS: econhdfe {__version__}")


if __name__ == "__main__":
    main()
