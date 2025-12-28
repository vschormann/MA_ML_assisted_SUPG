import pyvista as pv
import os
from dolfinx import fem, plot, io
from mpi4py import MPI

def plot_mesh(i):
    mesh_path = os.path.join("data/example_2_1/training_set/fem_data/", f"mesh_{i}.xdmf")
    with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as reader:
        mesh = reader.read_mesh()
    Wh = fem.functionspace(mesh, ('P', 1))
    c_topology, c_cell_types, c_geometry = plot.vtk_mesh(Wh)
    grid = pv.UnstructuredGrid(c_topology, c_cell_types, c_geometry)
    p.camera_position='xy'
    p.add_mesh(grid, show_edges=True)
    p.reset_camera()

p = pv.Plotter(notebook=True, shape=(2,2))



p.subplot(0,0)
p.add_text('1.')
plot_mesh(0)
p.zoom_camera(1.3)
p.subplot(0,1)
p.add_text('2.')
plot_mesh(4)
p.zoom_camera(1.3)
p.subplot(1,0)
p.add_text('3.')
plot_mesh(5)
p.zoom_camera(1.3)
p.subplot(1,1)
p.add_text('4.')
plot_mesh(6)
p.zoom_camera(1.3)
p.show()