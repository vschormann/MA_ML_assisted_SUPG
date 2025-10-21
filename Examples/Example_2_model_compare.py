from supg.sp_problems import pde_data1 as pde_data
from torch_classes.nn_models import md1, md2
from torch_classes.supg_torch import supg_loss
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyvista as pv
from supg import supg
import math
from dolfinx import mesh as msh
from mpi4py import MPI
from supg.tabulate_problem_data import tabulate_problem_data


domain = msh.create_unit_square(MPI.COMM_WORLD, 10, 10, msh.CellType.triangle)

sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)
sd.set_weights(1e-1)
stg = sd.create_stage()

device = torch.device("mps")
inpt = torch.tensor(np.array(tabulate_problem_data(domain, pde_data=pde_data)[1:2]).reshape(1,5,-1), dtype=torch.float, device=device)


model = md2(inpt.shape[1], inpt.shape[2]).to(device)

learning_rate = 1e-3


origin = torch.zeros_like(model(inpt))
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


regulization = torch.nn.MSELoss(reduction='sum')
for t in range(1500):
    pred = model(inpt)
    loss = supg_loss(sd, pred) + regulization(pred, origin)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

comp_model = md1(pred)
comp_sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)
supg_loss(comp_sd, pred)

comp_optimizer = torch.optim.SGD(comp_model.parameters(), lr=learning_rate)

filename = 'model_compare.mp4'

p = pv.Plotter(notebook=True, shape=(1,2))
p.open_movie(filename, framerate=30)

p.subplot(0,0)
stg.add_data(sd.uh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

p.subplot(0,1)
stg.add_data(comp_sd.uh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

p.write_frame()


for t in range(500):
    p.clear()
    pred = model(inpt)
    comp_pred = comp_model()

    loss = supg_loss(sd, pred) + regulization(pred, origin)
    comp_loss = supg_loss(comp_sd, comp_pred)

    p.subplot(0,0)
    stg.add_data(sd.uh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

    p.add_text('NN-model 1: 'f'Iterationen: {math.floor(t/10)*10}', name='time-label')
    p.subplot(0,1)
    p.add_text('Gradient descend')
    stg.add_data(comp_sd.uh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)
    p.reset_camera()
    p.write_frame()

    optimizer.zero_grad()
    comp_optimizer.zero_grad()

    loss.backward()
    comp_loss.backward()

    optimizer.step()
    comp_optimizer.step()

p.close()