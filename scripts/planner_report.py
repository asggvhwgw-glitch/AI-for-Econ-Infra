#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from econhdfe.planner import get_thread_calibration, write_planner_developer_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a privacy-minimized econhdfe planner report.")
    ap.add_argument("--output", type=Path, default=Path("econhdfe-planner-report.md"))
    ap.add_argument("--format", choices=("md", "json"), default=None)
    ap.add_argument("--force-calibration", action="store_true")
    args = ap.parse_args()
    calibration = get_thread_calibration(force=args.force_calibration)
    path = write_planner_developer_report(args.output, format=args.format, calibration=calibration)
    print(path)


if __name__ == "__main__":
    main()
