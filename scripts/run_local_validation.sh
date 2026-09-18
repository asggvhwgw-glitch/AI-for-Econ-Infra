#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/run_tests.py -- -q
python scripts/smoke_install.py --in-place
python benchmarks/bench_cpu_projection.py --quick
python benchmarks/ppml/bench_ppml.py --scenario two_way --n 50000 >/dev/null
