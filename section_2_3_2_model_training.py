from section_2_3_2_supervised_dataset import train_dataset, train_loader, test_loader, test_dataset
from example_2_1_dataloader import test_loader

from example_2_1_model import model

import numpy as np
import torch
import os
from dolfinx import io
from dolfinx import fem
from dolfinx import default_scalar_type
from dolfinx import mesh as msh
import ufl
import torch
from mpi4py import MPI

from utils.FEniCSx_PyTorch_interface import FEniCSx_PyTorch_interface
from utils.FEniCSx_solver import FEniCSx_solver



def train_loop(dataloader, nn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    nn.train()
    train_loss = 0
    loss_fn = torch.nn.MSELoss()
    for batch, (X, T) in enumerate(dataloader):
        agg_loss = 0
        batch_size = len(X)
        for idx in range(batch_size):
        # Compute prediction and loss
            z2 = nn(X[idx])
            t = T[idx]
            loss = loss_fn(z2, t)/batch_size
            loss.backward()
            agg_loss += loss.item()
        # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = agg_loss, batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss += agg_loss
    train_loss /= num_batches
    print(f"train_loss: {train_loss:>7f}")
    return(train_loss)


def test_loop(nn):
    size = len(test_dataset)
    test_loss = 0
    mesh_dir="data/example_2_1/test_set/fem_data/"
    input_dir="data/example_2_1/test_set/inputs/"

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
        
        input_path = os.path.join(input_dir, f"x_{idx}.pt")
        input = torch.load(input_path) 
        pred = nn(input)
        with torch.no_grad():
            # Compute prediction and loss
                test_loss += fem_solver(pred)

    test_loss /= size
    print(f"Test Error: {test_loss:>8f} \n")
    return(test_loss)


def train(model_dir, model_init, optim, dtype, device, epochs, train_loss_file, test_loss_file):
    nn = model(device=device, dtype=dtype, dir=model_init)
    optimizer = optim(nn.parameters())
    train_loss_arrray = np.array([])
    test_loss_arrray = np.array([])
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_arrray = np.append(train_loss_arrray, train_loop(train_loader, nn, optimizer))
        test_loss_arrray = np.append(test_loss_arrray,test_loop(nn))
        torch.save(nn.state_dict(), f'{model_dir}_{t}.pth')

    np.save(file=train_loss_file, arr=train_loss_arrray)

    np.save(file=test_loss_file, arr=test_loss_arrray)


input = train_dataset[0]['x']


train(
         model_dir='data/section_2_3_2/models/nn_supervised_Adam',
         model_init='data/example_2_1/models/nn_init.pth',
         optim=torch.optim.Adam,
         dtype=input.dtype,
         device=input.device,
         epochs=20,
         train_loss_file='data/section_2_3_2/supervised_train_loss_Adam.npy',
         test_loss_file='data/section_2_3_2/supervised_test_loss_Adam.npy'
         )


print("Done!")



