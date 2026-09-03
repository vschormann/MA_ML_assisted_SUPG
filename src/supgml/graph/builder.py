"""High-level construction of PyTorch Geometric data objects."""

from dataclasses import dataclass

import numpy as np

from .features import STANDARD_FEATURES, solver_node_features
from .topology import cell_adjacency, relative_edge_features


@dataclass(frozen=True)
class GraphBuilder:
    """Convert a configured SUPG solver into a documented graph schema."""

    include_edge_features: bool = True
    include_self_edges: bool = True

    def build(self, solver, target=None, problem_id=None, mesh_id=None, upper=None):
        import torch
        from dolfinx import mesh as msh
        from torch_geometric.data import Data

        edges = cell_adjacency(solver.domain, include_self=self.include_self_edges)
        attributes = {
            "x": torch.as_tensor(solver_node_features(solver), dtype=torch.float32),
            "edge_index": edges,
            "feature_names": STANDARD_FEATURES,
            "schema_version": "1",
        }
        if self.include_edge_features:
            index_map = solver.domain.topology.index_map(solver.domain.topology.dim)
            cell_count = index_map.size_local + index_map.num_ghosts
            cells = np.arange(cell_count, dtype=np.int32)
            points = msh.compute_midpoints(solver.domain, solver.domain.topology.dim, cells)[:, :2]
            attributes["edge_attr"] = torch.as_tensor(
                relative_edge_features(points, edges), dtype=torch.float32
            )
        if target is not None:
            attributes["y"] = torch.as_tensor(target, dtype=torch.float32).reshape(-1, 1)
        if upper is not None:
            attributes["upper"] = torch.as_tensor(upper, dtype=torch.float32).reshape(-1, 1)
        if problem_id is not None:
            attributes["prblm_id"] = int(problem_id)
        if mesh_id is not None:
            attributes["mesh_id"] = int(mesh_id)
        return Data(**attributes)
