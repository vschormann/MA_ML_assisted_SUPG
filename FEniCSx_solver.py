"""Compatibility imports for notebooks written before the package refactor."""

from supgml.fem import LinearSolver, interpolate_expr
from supgml.supg import (
    SUPG_grad_activation_solver,
    SUPG_grad_adjoint_method_solver,
    SUPG_solver,
    SaturatingLoss,
)
from supgml.viz import curve_plotter, fem_plotter_grid, plot_fn

loss_act_class = SaturatingLoss

__all__ = [
    "LinearSolver",
    "SUPG_solver",
    "SUPG_grad_adjoint_method_solver",
    "SUPG_grad_activation_solver",
    "curve_plotter",
    "fem_plotter_grid",
    "interpolate_expr",
    "loss_act_class",
    "plot_fn",
]
