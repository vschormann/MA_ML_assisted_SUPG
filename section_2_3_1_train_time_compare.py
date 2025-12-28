import time
from dolfinx import io, fem, mesh as msh, default_scalar_type
import ufl
import torch
import os
from mpi4py import MPI
import numpy as np
from utils.FEniCSx_solver import FEniCSx_solver
from utils.FEniCSx_PyTorch_interface import FEniCSx_PyTorch_interface

from example_2_1_model import model
from example_2_1_dataloader import test_dataset, test_loader

def train_loop(dataloader, nn, optimizer):
    nn.train()
    train_loss = 0
    for batch, (X, fem_solver) in enumerate(dataloader):
        agg_loss = 0
        batch_size = len(X)
        for idx in range(batch_size):
        # Compute prediction and loss
            z2 = nn(X[idx])
            loss = 1/batch_size*fem_solver[idx](z2)
            loss.backward()
            agg_loss += loss.item()
        # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

    return(train_loss)



start = time.perf_counter()
input = test_dataset[0]['x']
nn=model(dtype=input.dtype, device=input.device, dir='data/example_2_1/models/nn_init.pth')
optimizer = torch.optim.Adam(nn.parameters())

for i in range(20):
    train_loop(dataloader=test_loader, nn=nn, optimizer=optimizer)
t1 = (time.perf_counter() - start)
disp1 = f"example 2.1 model training on test set takes {t1:.3f} seconds"
print(disp1)



start = time.perf_counter()

dataset = test_dataset
iterations = 20


model_init='data/example_2_1/models/nn_init.pth'

nn= model(device='cpu', dtype=torch.float64, dir=model_init)
w = nn.state_dict()['w'].detach().numpy()

size = len(dataset)
mesh_dir = "data/example_2_1/test_set/fem_data/"
iterations = 20
lst = os.listdir(mesh_dir)
size = len([f for f in lst if f.endswith('.xdmf')])
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
    ex_exp = x[0]*(1-ufl.exp(-(1-x[0])/eps))* (1 - (((ufl.exp(-(1-x[1])/eps)  + ufl.exp(-(x[1])/eps)))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))


    exp = fem.Expression(ex_exp, Wh.element.interpolation_points())

    u_ex = fem.Function(Wh)
    u_ex.interpolate(exp)

    loss = (uh-u_ex)**2 * ufl.dx
    pde_data = mesh,Wh,uh,eps,b,None,f,None,bcs

    FEniCSx = FEniCSx_solver(pde_data=pde_data, loss_form=loss)

    lst = []
    bK = np.array([1,0])
    for v in mesh.geometry.x[:,0:2][mesh.geometry.dofmap]:
        xK = np.concat((v[2]-v[1], v[2]-v[0], v[1]-v[0], bK), axis=None)
        lst.append(xK)

    xnp=np.array(lst).T
    z1 = np.matmul(w, xnp)
    Pe = z1/(2*1e-8)
    supg_params = z1/2*(1/np.tanh(Pe)-(2*1e-8)*z1)

    FEniCSx.set_weights(supg_params)
    params = torch.tensor(supg_params, requires_grad=True)
    autograd_func = FEniCSx_PyTorch_interface.apply
    
    def fem_solver(weights):
        return autograd_func(weights, FEniCSx)
    
    optimizer = torch.optim.Adam([params])
    train_loss = 0
    for steps in range(iterations):
        optimizer.zero_grad()
        loss = fem_solver(params)
        loss.backward()
        # Backpropagation
        optimizer.step()
t2 = time.perf_counter() - start
disp2 = f"Section 2.3.1: performing 20 Adam steps for each mesh in the training set takes {t2:.3f} seconds"
print(disp2)

disp3 = f"Section 2.3.1: Computing 20*size_dataset gradients in one epoch can be done in 1/{int(t1/t2)} of the time it takes to compute size_dataset gradients in 20 epochs."
print(disp3)