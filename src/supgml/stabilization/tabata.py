"""Tabata upwinding helpers for triangular P1 meshes."""

import numpy as np
import scipy.sparse.linalg
import ufl
from dolfinx import fem, mesh as msh

from supgml.fem import interpolate_expr


def _assemble_matrix(form):
    compiled = fem.form(form)
    petsc_matrix = fem.create_matrix(compiled)
    matrix = petsc_matrix.to_scipy()
    fem.assemble_matrix(petsc_matrix, compiled)
    return matrix


def _upwind_cell(mesh, node, b1, b2, tolerance=1e-12):
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(0, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    adjacent = msh.compute_incident_entities(
        mesh.topology, np.array([node], dtype=np.int32), 0, tdim
    )
    coordinates = mesh.geometry.x
    origin = coordinates[node]
    direction = np.array([-b1.x.array[node], -b2.x.array[node]])
    cells_to_vertices = mesh.topology.connectivity(tdim, 0)
    for cell in adjacent:
        vertices = cells_to_vertices.links(cell)
        others = vertices[vertices != node]
        first = coordinates[others[0], :2] - origin[:2]
        second = coordinates[others[1], :2] - origin[:2]
        if np.cross(first, second) < 0:
            first, second = second, first
        if np.cross(first, direction) >= -tolerance and np.cross(direction, second) >= -tolerance:
            return cell
    return None


def _dof_maps(continuous_space, discontinuous_space, mesh):
    continuous_coordinates = continuous_space.tabulate_dof_coordinates()
    discontinuous_coordinates = discontinuous_space.tabulate_dof_coordinates()
    lookup = {tuple(np.round(row, 12)): i for i, row in enumerate(continuous_coordinates)}
    nodes = np.array([lookup[tuple(np.round(row, 12))] for row in discontinuous_coordinates])
    cells = np.zeros(len(discontinuous_space.dofmap.list.reshape(-1)), dtype=int)
    for cell in range(mesh.topology.index_map(2).size_global):
        dofs = fem.locate_dofs_topological(discontinuous_space, 2, np.array([cell]))
        cells[dofs] = cell
    return nodes, cells


def _local_dof(node, cell, node_map, cell_map):
    candidates = np.intersect1d(np.where(node_map == node), np.where(cell_map == cell))
    return candidates[0] if len(candidates) else None


def _upwinding_matrix(mesh, velocity, space, trial, test):
    discontinuous = fem.functionspace(mesh, ("DG", 1))
    ud = ufl.TrialFunction(discontinuous)
    vd = ufl.TestFunction(discontinuous)
    node_map, cell_map = _dof_maps(space, discontinuous, mesh)
    b1, b2 = interpolate_expr(velocity[0], space), interpolate_expr(velocity[1], space)
    matrix = _assemble_matrix((trial.dx(0) + trial.dx(1)) * test * ufl.dx)
    derivative_x = _assemble_matrix(ud.dx(0) * vd * ufl.dx)
    derivative_y = _assemble_matrix(ud.dx(1) * vd * ufl.dx)
    for node in range(mesh.topology.index_map(0).size_global):
        columns = matrix[node].indices
        matrix[node, columns] = 0
        cell = _upwind_cell(mesh, node, b1, b2)
        if cell is None:
            continue
        row = _local_dof(node, cell, node_map, cell_map)
        for column_node in columns:
            column = _local_dof(column_node, cell, node_map, cell_map)
            if column is not None:
                matrix[node, column_node] = (
                    b1.x.array[node] * derivative_x[row, column]
                    + b2.x.array[node] * derivative_y[row, column]
                )
    return matrix


def tabata(mesh, boundary_values, boundary_dofs, diffusion, velocity, source, reaction=None):
    """Compute the notebook's Tabata upwind approximation."""

    space = fem.functionspace(mesh, ("CG", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    matrix = _assemble_matrix(diffusion * trial * test * ufl.dx)
    matrix += _upwinding_matrix(mesh, velocity, space, trial, test)
    if reaction is not None:
        matrix += _assemble_matrix(reaction * trial * test * ufl.dx)
    source_values = interpolate_expr(source, space).x.array
    rhs = source_values * fem.assemble_vector(fem.form(test * ufl.dx)).array
    values = boundary_values.x.array if hasattr(boundary_values, "x") else boundary_values
    for dof in boundary_dofs:
        columns = matrix[dof].indices
        matrix[dof, columns] = 0
        matrix[dof, dof] = 1
        rhs[dof] = values[dof]
    return scipy.sparse.linalg.spsolve(matrix, rhs)
