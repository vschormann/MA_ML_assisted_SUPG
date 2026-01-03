from dolfinx import mesh as msh, fem, io, default_scalar_type
from mpi4py import MPI
import os
import torch
import numpy as np
import ufl
from mpi4py import MPI
import numpy as np
from utils.FEniCSx_solver import FEniCSx_solver
from utils.FEniCSx_PyTorch_interface import FEniCSx_PyTorch_interface
from example_2_1_model import model

mean = 0


def mesh_to_x(mesh, cell_ind_to_grid, epsK, bK, cK, fK, non_perturbed=False, device=None, dtype=None):
    lst = []
    for v in mesh.geometry.x[:,0:2][mesh.geometry.dofmap]:
        xK = np.concat((v, epsK, bK, cK, fK), axis=None)
        lst.append(xK)

    xnp=np.array(lst).T
    x = torch.tensor(xnp[:,cell_ind_to_grid], requires_grad=False, device=device, dtype=dtype)

    x = torch.nn.functional.pad(x, pad=(1,1,1,1))

    # left column: set v0, v1 to v0, v1 of neighboring column
    x[0:4,1:ny+1, 0] = x[0:4,1:ny+1, 1]
    # set collapsed vertices v2 and v3 to v1 at the same position
    x[4:6,1:ny+1, 0] = x[2:3,1:ny+1, 0]
    x[6:8,1:ny+1, 0] = x[2:3,1:ny+1, 0]


    # right column: set v0 to v2 of neighboring column
    x[0:4,1:ny+1, -1] = x[4:8,1:ny+1, -1 -1]
    # set collapsed vertices v2 and v3 to v1 at the same position
    x[4:6,1:ny+1, -1] = x[2:3,1:ny+1, -1]
    x[6:8,1:ny+1, -1] = x[2:3,1:ny+1, -1]


    # top row: set v0 to v0 of neighboring row
    x[0:2,0,1:nx+1] = x[2:4,1,1:nx+1]
    # top row: set v1 to v3 of neighboring row
    x[2:4,0,1:nx+1] = x[6:8,1,1:nx+1]
    # set collapsed vertices v2 and v3 to v1 at the same position
    x[4:6,0,1:nx+1] = x[2:3,0,1:nx+1]
    x[6:8,0,1:nx+1] = x[2:3,0,1:nx+1]

    # bottom row: set v0 to v0 of neighboring row
    x[0:2,-1,1:nx+1] = x[0:2,-1-1,1:nx+1]
    # bottom row: set v1 to v2 of neighboring row
    x[2:4,-1,1:nx+1] = x[4:6,-1-1,1:nx+1]
    # set collapsed vertices v2 and v3 to v1 at the same position
    x[4:6,-1,1:nx+1] = x[2:3,-1,1:nx+1]
    x[6:8,-1,1:nx+1] = x[2:3,-1,1:nx+1]

    # set upper left corner to v0 of neigboring horizontal edge
    x[0:2,0,0] = x[0:2,0,1]
    # set remaining vertices to v0
    x[2:4,0,0] = x[0:2,0,0]
    x[4:6,0,0] = x[0:2,0,0]
    x[6:8,0,0] = x[0:2,0,0]

    # set bottom left corner to v0 of neighboring horizontal edge
    x[0:2,-1,0] = x[0:2,-1,1]
    # set remaining vertices to v0
    x[2:4,-1,0] = x[0:2,-1,0]
    x[4:6,-1,0] = x[0:2,-1,0]
    x[6:8,-1,0] = x[0:2,-1,0]

    # set upper right corner to v1 of neighboring horizontal edge
    x[0:2,0,-1] = x[2:4,0,-1-1]
    # set remaining vertices to v0
    x[2:4,0,-1] = x[0:2,0,-1]
    x[4:6,0,-1] = x[0:2,0,-1]
    x[6:8,0,-1] = x[0:2,0,-1]

    # set bottom right corner to v1 of neighboring horizontal edge
    x[0:2,-1,-1] = x[2:4,-1,-1-1]
    # set remaining vertices to v0
    x[2:4,-1,-1] = x[0:2,-1,-1]
    x[4:6,-1,-1] = x[0:2,-1,-1]
    x[6:8,-1,-1] = x[0:2,-1,-1]

    # set the dirichlet boundary condition
    x[11,0,:] = 1
    x[11,-1,:] = 1
    x[11,:,0] = 1
    x[11,:,-1] = 1

    if non_perturbed:
        if np.all(bK == np.array([1,0])):
            x[12,0,:] = (x[0,0,:] + x[2,0,:])/2
            x[12,-1,:] = (x[0,-1,:] + x[2,-1,:])/2
            x[12,:,0] = (x[0,:,0] + x[2,:,0])/2
            x[12,:,-1] = (x[0,:,-1] + x[2,:,-1])/2
        if np.all(bK == np.array([-1,0])):
            x[12,0,:] = -(x[0,0,:] + x[2,0,:])/2
            x[12,-1,:] = -(x[0,-1,:] + x[2,-1,:])/2
            x[12,:,0] = -(x[0,:,0] + x[2,:,0])/2
            x[12,:,-1] = -(x[0,:,-1] + x[2,:,-1])/2
        if np.all(bK == np.array([0,1])):
            x[12,0,:] = (x[1,0,:] + x[3,0,:])/2
            x[12,-1,:] = (x[1,-1,:] + x[3,-1,:])/2
            x[12,:,0] = (x[1,:,0] + x[3,:,0])/2
            x[12,:,-1] = (x[1,:,-1] + x[3,:,-1])/2
        if np.all(bK == np.array([0,-1])):
            x[12,0,:] = -(x[1,0,:] + x[3,0,:])/2
            x[12,-1,:] = -(x[1,-1,:] + x[3,-1,:])/2
            x[12,:,0] = -(x[1,:,0] + x[3,:,0])/2
            x[12,:,-1] = -(x[1,:,-1] + x[3,:,-1])/2


    return x

model_init='data/example_2_1/models/nn_init.pth'

nn= model(device='cpu', dtype=torch.float64, dir=model_init)
w = nn.state_dict()['w'].detach().numpy()

def create_target_values(mesh, bK, non_perturbed=False):
    if non_perturbed:
        return torch.zeros(mesh.geometry.dofmap.shape[0])
    Wh = fem.functionspace(mesh, ('P', 2))
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = msh.exterior_facet_indices(mesh.topology)
    boundary_dofs = fem.locate_dofs_topological(Wh, mesh.topology.dim-1, boundary_facets)
    bcs = [fem.dirichletbc(fem.Constant(mesh, default_scalar_type(0.0)), boundary_dofs, Wh)]

    eps = fem.Constant(mesh, default_scalar_type(1e-8))

    b = ufl.as_vector((fem.Constant(mesh, default_scalar_type(bK[0])),fem.Constant(mesh, default_scalar_type(bK[1]))))
    f = fem.Constant(mesh, default_scalar_type(1.0))
    uh = fem.Function(Wh)

    x = ufl.SpatialCoordinate(mesh)
    if np.all(bK == np.array([1,0])) or np.all(bK == np.array([-1,0])):
        dir = 0
    else:
        dir = 1
    ex_exp = x[dir]*(1-ufl.exp(-(1-x[dir])/eps))* (1 - (((ufl.exp(-(1-x[(dir+1)%2])/eps)  + ufl.exp(-(x[(dir+1)%2])/eps)))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))

    if np.min(bK) < 0:
        ex_exp = (1-x[dir])*(1-ufl.exp(-(x[dir])/eps))* (1 - (((ufl.exp(-(1-x[(dir+1)%2])/eps)  + ufl.exp(-(x[(dir+1)%2])/eps)))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))

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

    train_loss += loss
    target = params
    return target



iterations = 500
comm = MPI.COMM_WORLD
epsK = 1e-8
cK = 0
fK = 1


num = 0

nx = 32
ny = 32
std = 1/(3*(nx+ny))

mesh = msh.create_unit_square(comm, nx, ny, msh.CellType.quadrilateral)
Yh = fem.functionspace(mesh, ('DG', 0))
pts = Yh.tabulate_dof_coordinates()
x = pts[:,0]
y = pts[:,1]
cell_ind_to_grid = np.lexsort((pts[:,0], -pts[:,1])).reshape(ny,nx)

vertices = mesh.geometry.x

cells = mesh.geometry.dofmap

element = mesh.geometry.cmap
mask= ~np.any((vertices[:,:2] == 0) | (vertices[:,:2] == 1), axis=1)
for bK in [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]:
    for non_perturbed in [True, False]:
        #with io.XDMFFile(mesh.comm, f"data/example_3_2/training_set/fem_data/mesh_{num}.xdmf", "w") as writer:
        #    writer.write_mesh(ms)
        x = mesh_to_x(mesh=mesh, cell_ind_to_grid=cell_ind_to_grid,epsK=epsK, bK=bK, cK=cK, fK=fK)
        torch.save(x, f'data/example_3_2/training_set/inputs/x_{num}.pt')
        t = create_target_values(mesh=mesh, bK=bK, non_perturbed=non_perturbed)
        torch.save(t, f'data/example_3_2/training_set/target_values/t_{num}.pt')
        num += 1

for i in range(200):
    for bK in [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]:
        for non_perturbed in [True, False]:
            noise = np.random.normal(mean, std, size=vertices.shape)
            noise[:,2] = 0
            new_vertices=vertices.copy()

            new_vertices[mask] += noise[mask]
            ms = msh.create_mesh(comm=comm, cells=cells, x=new_vertices, e=mesh.ufl_domain())
            
            #with io.XDMFFile(mesh.comm, f"data/example_3_2/training_set/fem_data/mesh_{num}.xdmf", "w") as writer:
            #    writer.write_mesh(ms)
            x = mesh_to_x(mesh=ms, cell_ind_to_grid=cell_ind_to_grid,epsK=epsK, bK=bK, cK=cK, fK=fK)
            torch.save(x, f'data/example_3_2/training_set/inputs/x_{num}.pt')
            t = create_target_values(mesh=ms, bK=bK, non_perturbed=non_perturbed)
            torch.save(t, f'data/example_3_2/training_set/target_values/t_{num}.pt')
            num += 1


num = 0

mesh = msh.create_unit_square(MPI.COMM_WORLD, nx, ny, msh.CellType.quadrilateral)

Yh = fem.functionspace(mesh, ('DG', 0))
pts = Yh.tabulate_dof_coordinates()
x = pts[:,0]
y = pts[:,1]
cell_ind_to_grid = np.lexsort((pts[:,0], -pts[:,1])).reshape(ny,nx)

vertices = mesh.geometry.x

cells = mesh.geometry.dofmap

element = mesh.geometry.cmap
mask= ~np.any((vertices[:,:2] == 0) | (vertices[:,:2] == 1), axis=1)
for i in range(50):
    for bK in [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]:
        for non_perturbed in [True, False]:
            noise = np.random.normal(mean, std, size=vertices.shape)
            noise[:,2] = 0
            new_vertices=vertices.copy()

            new_vertices[mask] += noise[mask]
            ms = msh.create_mesh(comm=comm, cells=cells, x=new_vertices, e=mesh.ufl_domain())
            #with io.XDMFFile(mesh.comm, f"data/example_3_2/training_set/fem_data/mesh_{num}.xdmf", "w") as writer:
            #    writer.write_mesh(ms)
            x = mesh_to_x(mesh=ms, cell_ind_to_grid=cell_ind_to_grid,epsK=epsK, bK=bK, cK=cK, fK=fK)
            torch.save(x, f'data/example_3_2/test_set/inputs/x_{num}.pt')
            t = create_target_values(mesh=ms, bK=bK, non_perturbed=non_perturbed)
            torch.save(t, f'data/example_3_2/test_set/target_values/t_{num}.pt')
            num += 1
