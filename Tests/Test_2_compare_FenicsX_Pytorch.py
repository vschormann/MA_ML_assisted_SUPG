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
sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=True)

sd.set_weights(1e-1)
model = nn_models.md1(torch.Tensor(sd.yh.x.array).view(1,-1))

weights = sd.yh.x.array

comp_loss_array = np.array([])
learning_rate = 1e-4

for t in range(1000):
    comp_loss_array = np.append(comp_loss_array, sd.global_loss())

    weights -= learning_rate * sd.constrained_grad()
    sd.set_weights(weights)
comp_vec = sd.uh.x.array


sd.set_weights(1e-1)

loss_array = np.array([])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for t in range(1000):
    pred = model()
    loss = supg_loss(sd, pred)
    loss_array = np.append(loss_array, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if np.allclose(loss_array, comp_loss_array):
    print('Same loss values')
else:
    print(np.abs(loss_array - comp_loss_array))
if np.allclose(sd.uh.x.array, comp_vec):
    print('Same result after optimization')
else:
    print(np.abs(sd.uh.x.array - comp_vec))


x = np.arange(len(loss_array))

plt.plot(x, loss_array, marker='o', linestyle='-')
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Pytorch loss')
plt.grid(True)
plt.show()