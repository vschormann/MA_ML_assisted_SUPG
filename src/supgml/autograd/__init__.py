"""Differentiable adapters around numerical solvers."""

from .fenics import FEniCSx_PyTorch_interface, batched_loss_fn, fem_solver

FEMObjective = fem_solver
BatchedFEMLoss = batched_loss_fn

__all__ = [
    "FEniCSx_PyTorch_interface",
    "fem_solver",
    "batched_loss_fn",
    "FEMObjective",
    "BatchedFEMLoss",
]
