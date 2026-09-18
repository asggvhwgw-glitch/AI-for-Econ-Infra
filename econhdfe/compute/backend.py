from __future__ import annotations
import numpy as np


def get_array_module(backend: str):
    if backend == "numpy":
        return np
    if backend == "cupy":
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError("backend='cupy' requires a matching CuPy CUDA build") from e
        return cp
    raise ValueError("backend must be 'numpy' or 'cupy'")
