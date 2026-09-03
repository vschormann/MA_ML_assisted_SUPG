"""Compatibility API for training notebooks created before the refactor.

Importing this legacy module retains the old dataset construction behavior.
New code should import from :mod:`supgml.data` and :mod:`supgml.training`.
"""

from torch_geometric.loader import DataLoader

from supgml.data import GraphDataset
from supgml.training import train

graph_dataset = GraphDataset
train_set = GraphDataset(root="data/training_set/input_values/")
test_set = GraphDataset(root="data/test_set/input_values/")


class train_loader(DataLoader):
    def __init__(self, batch_size, set=None, shuffle=True):
        dataset = set if set is not None else train_set
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle if set is None else False)


class test_loader(DataLoader):
    def __init__(self, batch_size):
        super().__init__(test_set, batch_size=batch_size)


__all__ = [
    "GraphDataset",
    "graph_dataset",
    "train",
    "train_set",
    "test_set",
    "train_loader",
    "test_loader",
]
