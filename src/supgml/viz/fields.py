"""Visualization of DOLFINx fields."""

import numpy as np
import pyvista
from dolfinx import fem, plot

from supgml.fem import sample_function


class FieldGrid:
    """A PyVista grid backed by a DOLFINx function space."""

    def __init__(self, function_space):
        topology, cell_types, geometry = plot.vtk_mesh(function_space)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    def add_data(self, values, point=True, name="u"):
        self.grid.point_data.clear()
        self.grid.cell_data.clear()
        array = values if isinstance(values, np.ndarray) else values.x.array.real
        target = self.grid.point_data if point else self.grid.cell_data
        alternate = self.grid.cell_data if point else self.grid.point_data
        try:
            target[name] = array
        except ValueError:
            alternate[name] = array
            self.grid = (
                self.grid.cell_data_to_point_data()
                if point
                else self.grid.point_data_to_cell_data()
            )
        self.grid.set_active_scalars(name, preference="point" if point else "cell")
        return self


fem_plotter_grid = FieldGrid


def curve_plotter(points, function):
    return sample_function(function, points)


def plot_fn(function, warp=True):
    """Display a DOLFINx scalar function in a PyVista window."""

    try:
        grid = FieldGrid(function.function_space)
    except RuntimeError:
        space = fem.functionspace(function.function_space.mesh, ("CG", 1))
        grid = FieldGrid(space)
    grid.add_data(function)
    plotter = pyvista.Plotter()
    if warp:
        plotter.add_mesh(grid.grid.warp_by_scalar(), show_edges=True)
    else:
        plotter.camera_position = "xy"
        plotter.add_mesh(grid.grid, show_edges=True)
    plotter.reset_camera()
    return plotter.show()
