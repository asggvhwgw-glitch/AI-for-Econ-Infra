"""Outcome-model estimators that consume the shared HDFE/compute layers."""
from .ols import olshdfe, reghdfe
from .ppml import ppmlhdfe, PPMLHDFE, PPMLConfig, PPMLResult
