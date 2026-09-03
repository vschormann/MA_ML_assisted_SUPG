"""Finite-element assembly and interpolation helpers."""

from .interpolation import interpolate_expr, sample_function
from .linear import LinearSolver

__all__ = ["LinearSolver", "interpolate_expr", "sample_function"]
