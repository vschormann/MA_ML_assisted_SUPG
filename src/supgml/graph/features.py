"""Named, versionable node features for SUPG prediction graphs."""

import numpy as np


STANDARD_FEATURES = (
    "diffusion",
    "velocity_x",
    "velocity_y",
    "reaction",
    "source",
    "cell_diameter",
    "standard_solution",
    "solution_dx",
    "solution_dy",
)


def solver_node_features(solver):
    """Build the nine node features used by the data-generation notebook."""

    import ufl
    from supgml.fem import interpolate_expr

    space = solver.Yh
    expressions = (
        solver.eps,
        solver.b[0],
        solver.b[1],
        solver.c,
        solver.f,
        ufl.CellDiameter(solver.uh.function_space.mesh),
        solver.uh,
        solver.uh.dx(0),
        solver.uh.dx(1),
    )
    return np.column_stack([interpolate_expr(expr, space).x.array for expr in expressions])
