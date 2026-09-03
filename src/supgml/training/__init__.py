"""Supervised and differentiable FEM training loops."""

from .loops import fit, self_supervised_train, train, train_epoch

__all__ = ["fit", "train", "train_epoch", "self_supervised_train"]
