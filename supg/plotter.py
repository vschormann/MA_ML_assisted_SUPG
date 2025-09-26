import pyvista
from dolfinx import plot
import numpy as np
from dolfinx import mesh as msh, fem



class fem_plotter_grid:
    def __init__(self, Wh):
        c_topology, c_cell_types, c_geometry = plot.vtk_mesh(Wh)
        self.grid = pyvista.UnstructuredGrid(c_topology, c_cell_types, c_geometry)

    def add_data(self, u, point=True):
        self.grid.point_data.clear()
        self.grid.cell_data.clear()

        if isinstance(u, np.ndarray):
            val = u
        else:
            val = u.x.array.real
        if point:
            try:
                self.grid.point_data['u'] = val
            except:
                self.grid.cell_data['u'] = val
                self.grid = self.grid.cell_data_to_point_data()
        else:
            try:
                self.grid.cell_data['u'] = val
            except:
                self.grid.point_data['u'] = val
                self.grid = self.grid.point_data_to_cell_data()
        
        if point:
            self.grid.set_active_scalars('u', preference='point')
        else:
            self.grid.set_active_scalars('u', preference='cell')


