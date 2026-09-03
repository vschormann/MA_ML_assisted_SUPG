"""Typed data passed to convection-diffusion solvers."""

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple


@dataclass
class ConvectionDiffusionProblem:
    """Finite-element description of a stationary convection-diffusion PDE."""

    mesh: Any
    solution_space: Any
    solution: Any
    diffusion: Any
    velocity: Any
    reaction: Optional[Any]
    source: Any
    neumann_source: Optional[Any]
    boundary_conditions: Iterable[Any]

    @classmethod
    def from_legacy_tuple(cls, values: Tuple[Any, ...]):
        if len(values) != 9:
            raise ValueError("legacy PDE data must contain exactly nine values")
        return cls(*values)

    def as_legacy_tuple(self):
        return (
            self.mesh,
            self.solution_space,
            self.solution,
            self.diffusion,
            self.velocity,
            self.reaction,
            self.source,
            self.neumann_source,
            self.boundary_conditions,
        )
