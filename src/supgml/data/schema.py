"""Metadata for graph tensors saved by the package."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GraphSchema:
    """Names and dimensions needed to interpret a saved graph."""

    node_features: Tuple[str, ...]
    edge_features: Tuple[str, ...] = ()
    version: str = "1"

    @property
    def node_channels(self):
        return len(self.node_features)

    @property
    def edge_channels(self):
        return len(self.edge_features)
