from dolfinx import fem, default_scalar_type, mesh as msh, io
import ufl
from mpi4py import MPI
import torch
from utils.FEniCSx_solver import FEniCSx_solver
from utils.FEniCSx_PyTorch_interface import FEniCSx_PyTorch_interface
from example_2_1_dataloader import test_dataset
import os
import numpy as np



SGD_test_loss = 'data/example_2_1/exact_test_loss_SGD.npy'
a0 = np.load(SGD_test_loss)[-1]

Adam_test_loss = 'data/example_2_1/exact_test_loss_Adam.npy'
a1 = np.load(Adam_test_loss)[-1]


size = len(test_dataset)
a2 = 0
mesh_dir="data/example_2_1/test_set/fem_data/"
target_dir="data/example_2_1/test_set/target_values/"

for idx in range(size):
    mesh_path = os.path.join(mesh_dir, f"mesh_{idx}.xdmf")
    with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as reader:
        mesh = reader.read_mesh()


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
    ex_exp = x[0]*(1-ufl.exp(-(1-x[0])/eps))* (1 - ((ufl.exp(-(1-x[1])/eps)  + ufl.exp(-(x[1])/eps))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))


    exp = fem.Expression(ex_exp, Wh.element.interpolation_points())

    u_ex = fem.Function(Wh)
    u_ex.interpolate(exp)

    loss = (uh-u_ex)**2 * ufl.dx
    pde_data = mesh,Wh,uh,eps,b,None,f,None,bcs

    FEniCSx = FEniCSx_solver(pde_data=pde_data, loss_form=loss)

    autograd_func = FEniCSx_PyTorch_interface.apply
    
    def fem_solver(weights):
            return autograd_func(weights, FEniCSx)
    
    target_path = os.path.join(target_dir, f"t_{idx}.pt")
    target = torch.load(target_path) 
    with torch.no_grad():
        # Compute prediction and loss
            a2 += fem_solver(target)

a2 /= size
print(f"Test Error: {a2:>8f} \n")

supervised_test_loss = 'data/section_2_3_2/supervised_test_loss_Adam.npy'
a3 = np.load(supervised_test_loss)[-1]

y = np.array([a0,a1,a2,a3])


import matplotlib.pyplot as plt

x = ['SGD', 'Adam', 'direct optimization', 'supervised']

plt.bar(x,y)
plt.grid()
plt.savefig('section_2_3_2_model_compare.png')
plt.show()
