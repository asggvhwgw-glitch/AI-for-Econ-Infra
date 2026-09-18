# Post-architecture validation evidence

This directory contains reproducible development evidence comparing the unified Execution Planner checkpoint with the immediately preceding Data Layer checkpoint.

- `model_consistency.py`: deterministic OLS/IV/PPML/IV-PPML model matrix.
- `hdfe_stress.py`: generic four-FE thread/performance stress harness.
- `model_*json`: pre-/post-planner deterministic snapshots.
- `hdfe_*json`: explicit-thread and auto-thread HDFE evidence.
- `hetero_*json`: repeated heterogeneous/block four-model runs.
- `complex_*json`: repeated complex event-study runs.
- `representation_operator_mix_fresh.json`: dense/block operator calibration.
- `planner_overhead_fresh.json`: pure planner-overhead measurement.
- `legacy_expected_check.json`: retained Python-fixture comparison and known stale PPML DoF note.
- `validation_summary.json`: machine-readable aggregate used by `POST_ARCHITECTURE_VALIDATION.md`.

Wall-time values are development measurements from a shared container, not release performance promises. Statistical parity is a strict gate; performance conclusions use repeated medians and are not based on a single timing observation.
