import os
import numpy as np
from dolfinx import io
from dolfinx import fem
from dolfinx import default_scalar_type
from dolfinx import mesh as msh
import ufl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mpi4py import MPI

from example_2_1_model import model
from utils.FEniCSx_solver import FEniCSx_solver
from utils.FEniCSx_PyTorch_interface import FEniCSx_PyTorch_interface

model_init='data/example_2_1/models/nn_init.pth'

nn= model(device='cpu', dtype=torch.float64, dir=model_init)
w = nn.state_dict()['w'].detach().numpy()


class Dataset_section_2_3_1(Dataset):
    def __init__(self, mesh_dir):
        self.mesh_dir = mesh_dir

    def __len__(self):
        return int(len(sorted(os.listdir(self.mesh_dir)))/2)

    def __getitem__(self, idx):
        mesh_path = os.path.join(self.mesh_dir, f"mesh_{idx}.xdmf")
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

        return {'params':params, 'fem_solver':fem_solver}
    
    
def collate_fn(batch):
    fem_solver  = [b['fem_solver'] for b in batch]
    params = [b['params'] for b in batch]
    return params, fem_solver

train_dataset = Dataset_section_2_3_1(mesh_dir="data/example_2_1/training_set/fem_data/")