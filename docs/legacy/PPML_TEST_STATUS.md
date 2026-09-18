# Test status — 0.1.0a1

Local status: **34 tests passing** on the current Linux/Python 3.13 container.

Coverage includes:

- no-FE Poisson parity against Statsmodels;
- explicit-dummy one-FE parity;
- replica/optimized coefficient and VCE invariance;
- offset/exposure;
- recursive singleton pruning;
- pre/post-absorption collinearity;
- robust and clustered covariance;
- exact FE-only separation;
- all translated upstream simplex internal matrix examples;
- the upstream public mixed-simplex nonexistence example;
- the upstream primer case where `fe + simplex` correctly does *not* replace
  ReLU for joint-FE separation;
- ReLU primer and fixed-weight LSMR path;
- optional post-IRLS mu separation;
- upstream-style standardization order/scaling and extreme regressor scaling;
- public-command iteration defaults (`maxiter=10000`, simplex maxiter=1000);
- changing-weight compiled projector vs fresh-projector parity for 2FE and 3FE;
- exact hierarchical canonicalization while keeping requested-FE DoF/VCE.

External gates not passed in this container: licensed-Stata golden comparison,
upstream 17 separation datasets, Stata pweight/fweight parity, cross-platform
Python matrix, and 10M+ PPML stress.
