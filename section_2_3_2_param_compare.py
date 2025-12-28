from dolfinx import io, fem, mesh as msh, plot, default_scalar_type
import ufl
from mpi4py import MPI
import pyvista as pv
import os

import torch

from utils.FEniCSx_solver import FEniCSx_solver
from example_2_1_model import model


def plot_params(mesh, params):
    Wh = fem.functionspace(mesh, ('P', 2))
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = msh.exterior_facet_indices(mesh.topology)
    boundary_dofs = fem.locate_dofs_topological(Wh, mesh.topology.dim-1, boundary_facets)
    bcs = [fem.dirichletbc(fem.Constant(mesh, default_scalar_type(0.0)), boundary_dofs, Wh)]

    eps = fem.Constant(mesh, default_scalar_type(1e-8))
    b = ufl.as_vector((fem.Constant(mesh, default_scalar_type(1.0)),fem.Constant(mesh, default_scalar_type(0.0))))
    f = fem.Constant(mesh, default_scalar_type(1.0))
    uh = fem.Function(Wh)

    x = ufl.SpatialCoordinate(mesh)
    ex_exp = x[0]*(1-ufl.exp(-(1-x[0])/eps))* (1 - (((ufl.exp(-(1-x[1])/eps)  + ufl.exp(-(x[1])/eps)))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))


    exp = fem.Expression(ex_exp, Wh.element.interpolation_points())

    u_ex = fem.Function(Wh)
    u_ex.interpolate(exp)

    loss = (uh-u_ex)**2 * ufl.dx
    pde_data = mesh,Wh,uh,eps,b,None,f,None,bcs




    c_topology, c_cell_types, c_geometry = plot.vtk_mesh(Wh)

    fs = FEniCSx_solver(pde_data=pde_data, loss_form=loss)

    fs.set_weights(params)
    fem_grid = pv.UnstructuredGrid(c_topology, c_cell_types, c_geometry)
    fem_grid.cell_data[''] = fs.yh.x.array
    fem_grid.cell_data_to_point_data()
    return fem_grid


idx = 3

target = torch.load(f'data/example_2_1/training_set/target_values/t_{idx}.pt').detach().numpy()
input = torch.load(f'data/example_2_1/training_set/inputs/x_{idx}.pt')

with io.XDMFFile(MPI.COMM_WORLD, f'data/example_2_1/training_set/fem_data/mesh_{idx}.xdmf', "r") as reader:
    mesh = reader.read_mesh()

nn = model(device=input.device, dtype=input.dtype, dir='data/example_2_1/models/nn_exact_Adam_19.pth')

pred = nn(input).detach().numpy()





p = pv.Plotter(shape=(1,2))

p.subplot(0,0)
p.add_mesh(plot_params(mesh, target), show_edges=False, scalar_bar_args={"vertical":False}, clim=(0,1))

p.camera_position = 'xy'
p.zoom_camera(1.5)
p.show_axes()
p.add_text('Individually optimized')

p.subplot(0,1)
p.add_mesh(plot_params(mesh, pred), show_edges=False)

p.camera_position = 'xy'
p.zoom_camera(1.5)
p.show_axes()
p.add_text('Model output')


p.show()