"""Classical stabilization parameters and algebraic flux correction."""

from .afc import BJKAFC, KuzminAFC, ConvergenceMonitor, F_AFC_BJK, F_AFC_Kuzmin
from .parameters import directional_diameter, standard_tau, yh_std


def tabata(*args, **kwargs):
    """Lazily import and run the DOLFINx-backed Tabata solver."""

    from .tabata import tabata as solve

    return solve(*args, **kwargs)

__all__ = [
    "BJKAFC",
    "KuzminAFC",
    "ConvergenceMonitor",
    "F_AFC_BJK",
    "F_AFC_Kuzmin",
    "directional_diameter",
    "standard_tau",
    "yh_std",
    "tabata",
]
