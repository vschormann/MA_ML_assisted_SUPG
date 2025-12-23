import random
from supg import supg
from mpi4py import MPI
from dolfinx import mesh as msh
from dolfinx import fem
from dolfinx import default_scalar_type
import ufl
import pyvista as pv
import torch_classes.nn_models as nn_models
import torch
from torch_classes.supg_torch import supg_loss
import numpy as np
import matplotlib.pyplot as plt

lr=1e-4

def pde_data(domain):
    Wh = fem.functionspace(domain, ('P', 2))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = msh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(Wh, domain.topology.dim-1, boundary_facets)
    eps_val=1e-8+random.normalvariate(mu=0, sigma=1e-9)
    b1_val=1+random.normalvariate(mu=0, sigma=1e-3)
    b2_val=random.normalvariate(mu=0, sigma=1e-3)
    c_val=0
    f_val=1
    eps = fem.Constant(domain, default_scalar_type(eps_val))
    b = ufl.as_vector((fem.Constant(domain, default_scalar_type(b1_val)),fem.Constant(domain, default_scalar_type(b2_val))))
    c = fem.Constant(domain, default_scalar_type(c_val))
    f = fem.Constant(domain, default_scalar_type(f_val))
    bcs = [fem.dirichletbc(fem.Constant(domain, default_scalar_type(0.0)), boundary_dofs, Wh)]
    return Wh,eps,b,c,f,bcs,None

domain = msh.create_unit_square(MPI.COMM_WORLD, 20, 20, msh.CellType.triangle)

sd=supg.data(domain, pde_data, boundary_eval=False)
model = nn_models.md1(torch.Tensor(sd.yh.x.array).view(1,-1))
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
def closure():
    optimizer.zero_grad()
    pred = model()
    loss = supg_loss(sd, pred)
    loss.backward()
    return loss

sd2 = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)
model2 = nn_models.md1(torch.Tensor(sd2.yh.x.array).view(1,-1))
optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr)



sd3 = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)
model3 = nn_models.md1(torch.Tensor(sd3.yh.x.array).view(1,-1))
optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr)


loss_array = np.array([])
loss_array2 = np.array([])
loss_array3 = np.array([])

for t in range(200):
    loss_array = np.append(loss_array, optimizer.step(closure).item())

    pred2 = model2()
    loss2 = supg_loss(sd2, pred2)
    loss_array2 = np.append(loss_array2, loss2.item())
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    pred3 = model3()
    loss3 = supg_loss(sd3, pred3)
    loss_array3 = np.append(loss_array3, loss3.item())
    optimizer3.zero_grad()
    loss3.backward()
    optimizer3.step()

x = np.arange(len(loss_array))
plt.plot(x, loss_array, label= 'LBFGS', linestyle='-',)
plt.plot(x, loss_array2,label='SGD', linestyle='-')
#plt.plot(x, loss_array3,label='Adam', linestyle='-')
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Pytorch loss')
plt.legend()
plt.grid(True)
plt.show()