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
from .revised import RevisedGATv2, RevisedMLP, combined_supervised_loss

__all__ = [
    "GraphRegressor",
    "create_model",
    "BoundedOutput",
    "NonNegativeOutput",
    "RevisedGATv2",
    "RevisedMLP",
    "combined_supervised_loss",
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
