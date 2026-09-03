"""Objective transformations used by adjoint SUPG optimization."""

import numpy as np


class SaturatingLoss:
    """Smoothly increase to one and remain constant above ``threshold``."""

    def __init__(self, threshold):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self.t0 = threshold

    def _scaled(self, values):
        return np.asarray(values) / self.threshold

    def __call__(self, values):
        values = np.asarray(values)
        z = self._scaled(values)
        return np.where(
            values > self.threshold,
            1,
            0.5 * z**4 - z**3 - 0.5 * z**2 + 2 * z,
        )

    def derivative(self, values):
        values = np.asarray(values)
        z = self._scaled(values)
        return np.where(
            values > self.threshold,
            0,
            (2 * z**3 - 3 * z**2 - z + 2) / self.threshold,
        )

    dx = derivative
