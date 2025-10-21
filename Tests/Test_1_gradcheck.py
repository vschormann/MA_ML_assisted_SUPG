from supg.sp_problems import pde_data1 as pde_data
from torch_classes.supg_torch import supg_loss
import torch
from supg import supg
import numpy as np
import dolfinx.mesh as msh
import mpi4py.MPI as MPI

domain = msh.create_unit_square(MPI.COMM_WORLD, 16, 16, msh.CellType.triangle)
sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=True)
dtype = torch.double
device = torch.device('cpu')


def test_loss(weights):
    return supg_loss(sd, weights)

size = sd.yh.x.index_map.size_local

for i in range(10):
    weights = np.random.rand(size)
    sd.set_weights(weights=weights)
    test = torch.tensor(sd.yh.x.array[:size], dtype=dtype, device=device, requires_grad=True).view(1, -1)

    print(str(i) + ': ' + str(torch.autograd.gradcheck(test_loss, (test), eps=1e-4, atol=1e-2, raise_exception=False)))
