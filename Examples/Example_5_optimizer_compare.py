from supg.sp_problems import pde_data1 as pde_data
from supg import supg
from torch_classes.supg_torch import supg_loss
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_classes.nn_models as nn_models
import dolfinx.mesh as msh
import mpi4py.MPI as MPI


domain = msh.create_unit_square(MPI.COMM_WORLD, 16, 16, msh.CellType.triangle)
learning_rate = 1e-4


loss_array = np.array([])
sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)
sd.set_weights(1e-1)
model = nn_models.md1(torch.Tensor(sd.yh.x.array).view(1,-1))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for t in range(1000):
    pred = model()
    loss = supg_loss(sd, pred)
    loss_array = np.append(loss_array, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


comp_loss_array = np.array([])
comp_sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)
comp_sd.set_weights(1e-1)
comp_model = nn_models.md1(torch.Tensor(comp_sd.yh.x.array).view(1,-1))
comp_optimizer = torch.optim.Adam(comp_model.parameters(), lr=learning_rate)
for t in range(1000):
    comp_pred = comp_model()
    comp_loss = supg_loss(comp_sd, comp_pred)
    comp_loss_array = np.append(comp_loss_array, comp_loss.item())

    comp_optimizer.zero_grad()
    comp_loss.backward()
    comp_optimizer.step()


x = np.arange(len(loss_array))

plt.plot(x, loss_array, label= 'SGD', linestyle='-',)
plt.plot(x, comp_loss_array,label='Adam', linestyle='-')
#plt.yscale('log')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('optimizer compare  loss')
plt.grid(True)
plt.show()
