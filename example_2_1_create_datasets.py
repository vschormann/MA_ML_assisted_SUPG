from dolfinx import mesh as msh
from dolfinx import io
import torch
from mpi4py import MPI
import numpy as np

rng = np.random.default_rng()

def mesh_to_x(mesh, bK, device=None, dtype=None):
    lst = []
    for v in mesh.geometry.x[:,0:2][mesh.geometry.dofmap]:
        xK = np.concat((v[2]-v[1], v[2]-v[0], v[1]-v[0], bK), axis=None)
        lst.append(xK)

    xnp=np.array(lst).T
    return torch.tensor(xnp, requires_grad=False, device=device, dtype=dtype)

bK = np.array([1,0])

num = 0
# base meshes
for s in range(4):
    domain = msh.create_unit_square(MPI.COMM_WORLD, int(2**(s+2)), int(2**(s+2)), msh.CellType.triangle)
    with io.XDMFFile(domain.comm, f"data/example_2_1/training_set/fem_data/mesh_{num}.xdmf", "w") as writer:
        writer.write_mesh(domain)
    x = mesh_to_x(domain, bK)
    torch.save(x, f'data/example_2_1/training_set/inputs/x_{num}.pt')
    num += 1

for j in range(10): #sample 10 times overall
    for i in range(1,8): #iterate through 1/8, 2/8,..., 7/8 of al edges in the current mesh 
        for s in range(3): #iterate through 4 by 4, 8 by 8, 16 by 16 base-mesh
            domain = msh.create_unit_square(MPI.COMM_WORLD, int(2**(s+2)), int(2**(s+2)), msh.CellType.triangle)
            for k in range(3-s): #refine so that no edge is smaller than in a 32 by 32 mesh
                domain.topology.create_entities(1)
                N = domain.topology.index_map(1).size_local
                edges = rng.choice(N, size=int(N*i/8), replace=False)
                domain = msh.refine(domain, edges)[0]
                with io.XDMFFile(domain.comm, f"data/example_2_1/training_set/fem_data/mesh_{num}.xdmf", "w") as writer:
                    writer.write_mesh(domain)
                x = mesh_to_x(domain, bK)
                torch.save(x, f'data/example_2_1/training_set/inputs/x_{num}.pt')
                num += 1


num = 0
for j in range(2): #sample 2 times overall
    for i in range(1,8): #iterate through 1/8, 2/8,..., 7/8 of al edges in the current mesh 
        for s in range(3): #iterate through 4 by 4, 8 by 8, ..., 32 by 32 base-mesh
            domain = msh.create_unit_square(MPI.COMM_WORLD, int(2**(s+2)), int(2**(s+2)), msh.CellType.triangle)
            for k in range(3-s): #refine so that eno edge is smaller than in a 64 by 64 mesh
                domain.topology.create_entities(1)
                N = domain.topology.index_map(1).size_local
                edges = rng.choice(N, size=int(N*i/8), replace=False)
                domain = msh.refine(domain, edges)[0]
                with io.XDMFFile(domain.comm, f"data/example_2_1/test_set/fem_data/mesh_{num}.xdmf", "w") as writer:
                    writer.write_mesh(domain)
                x = mesh_to_x(domain, bK)
                torch.save(x, f'data/example_2_1/test_set/inputs/x_{num}.pt')
                num += 1