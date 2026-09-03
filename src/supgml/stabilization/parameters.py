"""SUPG parameter formulas and mesh-size measurements."""

import numpy as np


def standard_tau(cell_diameter, diffusion=1e-8, velocity_norm=1.0):
    """Compute the standard cellwise SUPG parameter.

    The limiting value for zero Péclet number is evaluated safely instead of
    producing a division-by-zero warning.
    """

    h = np.asarray(cell_diameter, dtype=float)
    epsilon = np.asarray(diffusion, dtype=float)
    speed = np.asarray(velocity_norm, dtype=float)
    if np.any(epsilon <= 0):
        raise ValueError("diffusion must be positive")
    if np.any(speed < 0):
        raise ValueError("velocity_norm cannot be negative")
    peclet = h * speed / (2 * epsilon)
    xi = np.zeros_like(peclet, dtype=float)
    nonzero = np.abs(peclet) > 1e-7
    xi[nonzero] = 1 / np.tanh(peclet[nonzero]) - 1 / peclet[nonzero]
    xi[~nonzero] = peclet[~nonzero] / 3
    denominator = 2 * speed
    return np.divide(h * xi, denominator, out=np.zeros_like(h), where=denominator != 0)


yh_std = standard_tau


def directional_diameter(mesh, direction):
    """Return each triangular cell's diameter parallel to ``direction``."""

    from dolfinx import fem, mesh as msh

    tdim = mesh.topology.dim
    if tdim != 2:
        raise ValueError("directional_diameter supports two-dimensional meshes")
    mesh.topology.create_connectivity(tdim, 0)
    cells_to_vertices = mesh.topology.connectivity(tdim, 0)

    vector_space = fem.functionspace(mesh, ("DG", 0, (2,)))
    field = fem.Function(vector_space)
    field.interpolate(fem.Expression(direction, vector_space.element.interpolation_points()))
    directions = field.x.array.reshape((-1, 2))

    output_space = fem.functionspace(mesh, ("DG", 0))
    output = fem.Function(output_space)
    index_map = mesh.topology.index_map(tdim)
    cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    barycenters = msh.compute_midpoints(mesh, tdim, cells)
    values = np.zeros(len(cells), dtype=float)

    for cell in cells:
        point = barycenters[cell][:2]
        vector = directions[cell]
        norm = np.linalg.norm(vector)
        if norm < 1e-14:
            continue
        vector = vector / norm
        vertices = mesh.geometry.x[cells_to_vertices.links(cell)][:, :2]
        intersections = []
        for index in range(3):
            start = vertices[index]
            edge = vertices[(index + 1) % 3] - start
            system = np.column_stack((vector, -edge))
            if abs(np.linalg.det(system)) < 1e-14:
                continue
            distance, position = np.linalg.solve(system, start - point)
            if -1e-12 <= position <= 1 + 1e-12:
                intersections.append(distance)
        if len(intersections) >= 2:
            values[cell] = np.max(intersections) - np.min(intersections)
    output.x.array[:] = values
    return output
