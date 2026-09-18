# Upstream references

Primary implementation reference:

- Sergio Correia, Paulo Guimarães, Thomas Zylkin, `ppmlhdfe` Stata package.
- Correia, Guimarães & Zylkin (2020), “Fast Poisson Estimation with
  High-Dimensional Fixed Effects”, *Stata Journal* 20(1):95–115.
- Correia, Guimarães & Zylkin (2019), “Verifying the Existence of Maximum
  Likelihood Estimates for Generalized Linear Models”.

This package uses the upstream implementation as a behavioral oracle for IRLS,
separation and output validation, while maintaining an independent Python
module architecture. See `NOTICE.md` and `LICENSE`.
