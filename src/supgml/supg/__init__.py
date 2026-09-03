"""SUPG solvers and objective transformations."""

from .objectives import SaturatingLoss
from .problem import ConvectionDiffusionProblem
from .solver import (
    ActivatedAdjointSUPGSolver,
    AdjointSUPGSolver,
    SUPGSolver,
    SUPG_grad_activation_solver,
    SUPG_grad_adjoint_method_solver,
    SUPG_solver,
)

__all__ = [
    "ConvectionDiffusionProblem",
    "SaturatingLoss",
    "SUPGSolver",
    "AdjointSUPGSolver",
    "ActivatedAdjointSUPGSolver",
    "SUPG_solver",
    "SUPG_grad_adjoint_method_solver",
    "SUPG_grad_activation_solver",
]
