"""Sensitivity-derived graph connections and features."""

import numpy as np


def finite_difference_sensitivity(solver, cell, baseline=None, delta=1e-8, threshold=1e-8):
    """Estimate how one SUPG parameter affects the cellwise solution.

    Returns the derivative and a Boolean influence mask. The solver is restored
    to its baseline weights before returning.
    """

    from dolfinx import fem

    weights = np.array(solver.yh.x.array if baseline is None else baseline, copy=True)
    cell_function = fem.Function(solver.Yh)
    try:
        solver.set_weights(weights)
        cell_function.interpolate(solver.uh)
        initial = cell_function.x.array.copy()
        perturbed = weights.copy()
        perturbed[cell] += delta
        solver.set_weights(perturbed)
        cell_function.interpolate(solver.uh)
        derivative = (cell_function.x.array - initial) / delta
    finally:
        solver.set_weights(weights)
    return derivative, np.abs(derivative) > threshold
