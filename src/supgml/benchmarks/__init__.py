"""Named benchmark convection-diffusion problems."""

from .problems import (
    I_cross,
    bump,
    curved_wall,
    curved_wave,
    curved_waves,
    cylinder,
    falloff,
    hemker,
    lifted_edge,
    wedge,
)
from .registry import BENCHMARKS, create, int_to_prblm

__all__ = [
    "BENCHMARKS",
    "create",
    "int_to_prblm",
    "I_cross",
    "wedge",
    "bump",
    "lifted_edge",
    "cylinder",
    "falloff",
    "curved_wall",
    "curved_wave",
    "curved_waves",
    "hemker",
]
