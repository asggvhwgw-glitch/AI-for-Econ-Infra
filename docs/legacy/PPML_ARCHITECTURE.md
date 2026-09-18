# Architecture

The design rule is that statistical stages do not know how lower-level
numerical kernels are implemented.

- `api.py`: input validation and reusable DataFrame facade.
- `estimator.py`: one estimation transaction: sample -> separation -> IRLS ->
  final solve -> DoF/VCE -> result.
- `irls.py`: Poisson IRLS state machine only.
- `fe.py`: categorical FE representation, sample reductions, DoF and
  canonicalization.
- `projection.py`: mutable-weight compiled projectors; owns all IRLS topology
  reuse.
- `separation.py`: sequencing only.
- `separation_simplex.py`: independent mixed-X/FE simplex detector.
- `separation_relu.py`: independent iterative-rectifier detector.
- `standardize.py`: upstream-compatible numerical scaling/rescaling.
- `vce.py`: PPML score/bread adapter into the shared pyreghdfe VCE engine.
- `linalg.py`: small dense linear-algebra helpers.
- `config.py` / `results.py`: plain configuration and result contracts.

`engine="replica"` and `engine="optimized"` meet at the same estimator and IRLS
interfaces. Optimizations are selected only inside FE compilation/projection,
which makes A/B parity tests straightforward and prevents solver-specific
branches from spreading through the estimator.
