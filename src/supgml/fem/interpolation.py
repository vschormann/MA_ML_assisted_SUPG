"""DOLFINx interpolation and point-sampling utilities."""

import numpy as np
from dolfinx import default_scalar_type, fem
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points


def interpolate_expr(expr, function_space):
    """Interpolate an expression, treating ``None`` as zero."""

    result = fem.Function(function_space)
    if expr is None:
        expr = fem.Constant(function_space.mesh, default_scalar_type(0))
    compiled = fem.Expression(expr, function_space.element.interpolation_points())
    result.interpolate(compiled)
    return result


def sample_function(function, points):
    """Evaluate a DOLFINx function at points, returning NaN outside the mesh."""

    mesh = function.function_space.mesh
    points = np.asarray(points)
    tree = bb_tree(mesh, mesh.topology.dim)
    candidates = compute_collisions_points(tree, points)
    cells = compute_colliding_cells(mesh, candidates, points)
    values = np.full(len(points), np.nan)
    for index, point in enumerate(points):
        if len(cells.links(index)):
            values[index] = function.eval(point, cells.links(index)[0])
    return values
