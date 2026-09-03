"""Efficient graph topology and geometric edge features."""

import numpy as np


def cell_adjacency(mesh, include_self=True):
    """Return directed edges between cells that share at least one vertex.

    This uses DOLFINx connectivity rather than comparing every cell pair.
    """

    import torch

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_connectivity(0, tdim)
    cells_to_vertices = mesh.topology.connectivity(tdim, 0)
    vertices_to_cells = mesh.topology.connectivity(0, tdim)
    index_map = mesh.topology.index_map(tdim)
    cell_count = index_map.size_local + index_map.num_ghosts
    edges = set()
    for source in range(cell_count):
        if include_self:
            edges.add((source, source))
        for vertex in cells_to_vertices.links(source):
            for target in vertices_to_cells.links(vertex):
                if include_self or source != target:
                    edges.add((source, int(target)))
    ordered = sorted(edges)
    return torch.tensor(ordered, dtype=torch.long).T.contiguous()


def relative_position(source, target):
    """Return distance and normalized displacement from target to source."""

    displacement = np.asarray(source, dtype=float) - np.asarray(target, dtype=float)
    distance = np.linalg.norm(displacement)
    direction = np.divide(
        displacement,
        distance,
        out=np.zeros_like(displacement),
        where=distance > 0,
    )
    return np.concatenate(([distance], direction))


def relative_edge_features(points, edge_index):
    """Compute relative-position features for every directed edge."""

    source, target = edge_index.cpu().numpy()
    return np.stack([relative_position(points[s], points[t]) for s, t in zip(source, target)])
