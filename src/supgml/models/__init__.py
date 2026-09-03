"""Configurable graph models and output constraints."""

from .constraints import BoundedOutput, NonNegativeOutput
from .factory import GraphRegressor, create_model
from .legacy import (
    AbsRestriction,
    ClampRestriction,
    DirOpt,
    GAT,
    GATv2,
    GCN,
    MIX,
    MLP,
    PenaltyRestriction,
    SAGE,
    SigmoidRestriction,
    mha,
)

__all__ = [
    "GraphRegressor",
    "create_model",
    "BoundedOutput",
    "NonNegativeOutput",
    "MLP",
    "GCN",
    "SAGE",
    "GAT",
    "GATv2",
    "MIX",
    "mha",
    "AbsRestriction",
    "ClampRestriction",
    "SigmoidRestriction",
    "PenaltyRestriction",
    "DirOpt",
]
