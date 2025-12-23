import os
from dolfinx import io
from dolfinx import fem
from dolfinx import default_scalar_type
from dolfinx import mesh as msh
import ufl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mpi4py import MPI

from utils.FEniCSx_PyTorch_interface import FEniCSx_PyTorch_interface
from utils.FEniCSx_solver import FEniCSx_solver

batch_size = 10

class Dataset_example_2_1(Dataset):
    def __init__(self, input_dir, mesh_dir):
        self.input_dir = input_dir
        self.mesh_dir = mesh_dir

    def __len__(self):
        lst = os.listdir(self.input_dir)
        return len([f for f in lst if f.endswith('.pt')])

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, f"x_{idx}.pt")
        input = torch.load(input_path) 

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
        ex_exp = x[0]*(1-ufl.exp(-(1-x[0])/eps))* (1 - (((ufl.exp(-(1-x[1])/eps)  + ufl.exp(-(x[1])/eps)))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))


        exp = fem.Expression(ex_exp, Wh.element.interpolation_points())

        u_ex = fem.Function(Wh)
        u_ex.interpolate(exp)

        loss = (uh-u_ex)**2 * ufl.dx
        pde_data = mesh,Wh,uh,eps,b,None,f,None,bcs

        FEniCSx = FEniCSx_solver(pde_data=pde_data, loss_form=loss)
        autograd_func = FEniCSx_PyTorch_interface.apply
        def fem_solver(weights):
            return autograd_func(weights, FEniCSx)

        return {'x':input, 'fem_solver':fem_solver}
    
def collate_fn(batch):
    X = [b['x'] for b in batch]
    fem_solver  = [b['fem_solver'] for b in batch]
    return X, fem_solver

train_dataset = Dataset_example_2_1(input_dir="data/example_2_1/training_set/inputs/", mesh_dir="data/example_2_1/training_set/fem_data/")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

test_dataset = Dataset_example_2_1(input_dir="data/example_2_1/test_set/inputs/", mesh_dir="data/example_2_1/test_set/fem_data/")
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)