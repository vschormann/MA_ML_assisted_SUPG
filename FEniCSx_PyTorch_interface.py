"""Compatibility layer for differentiable FEM notebook imports."""

from supgml.autograd import FEniCSx_PyTorch_interface, batched_loss_fn, fem_solver
from supgml.data import CaseRepository, Data_to_solver
from supgml.training import self_supervised_train


class LegacyBatchedLoss(batched_loss_fn):
    """Construct the original training-set-backed loss on demand."""

    def __init__(self):
        from Training_utils import train_set

        configured = batched_loss_fn.from_repository(
            CaseRepository(), range(len(train_set)), split="train"
        )
        super().__init__(configured.fsl)


# The old zero-argument constructor remains available only through this shim.
batched_loss_fn = LegacyBatchedLoss

__all__ = [
    "FEniCSx_PyTorch_interface",
    "fem_solver",
    "batched_loss_fn",
    "Data_to_solver",
    "self_supervised_train",
]
