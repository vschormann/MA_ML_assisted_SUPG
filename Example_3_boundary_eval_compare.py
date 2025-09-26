from supg.sp_problems import dat1, stg
from torch_classes.nn_models import md1
from torch_classes.supg_torch import supg_loss
import torch
import pyvista as pv
from supg import supg

device = torch.device("mps")

sd = supg.data(*dat1, boundary_eval=False)
comp_sd = supg.data(*dat1, boundary_eval=True)

sd.set_weights(1e-1)
comp_sd.set_weights(1e-1)

model = md1(torch.Tensor(sd.yh.x.array).view(1,-1))
comp_model = md1(torch.Tensor(comp_sd.yh.x.array).view(1,-1))

learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
comp_optimizer = torch.optim.SGD(comp_model.parameters(), lr=learning_rate)

filename = 'boundary_eval_compare.mp4'

p = pv.Plotter(notebook=True, shape=(2,2))
p.open_movie(filename, framerate= 40)

p.subplot(0,0)
stg.add_data(sd.uh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)
p.add_text('No boundary_eval')


p.subplot(0,1)
p.add_text('With boundary_eval')
stg.add_data(comp_sd.uh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)


p.subplot(1,0)
stg.add_data(sd.yh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)
p.add_text('Weights no boundary_eval')


p.subplot(1,1)
p.add_text('Weights with boundary_eval')
stg.add_data(comp_sd.yh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)
p.write_frame()


for t in range(200):
    p.clear()
    pred = model()
    comp_pred = comp_model()

    loss = supg_loss(sd, pred)
    comp_loss = supg_loss(comp_sd, comp_pred)

    p.subplot(0,0)
    stg.add_data(sd.uh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)
    p.add_text('No boundary_eval: 'f'Iteration: {t}', name='time-label')

    p.subplot(0,1)
    p.add_text('With boundary_eval')
    stg.add_data(comp_sd.uh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

    p.subplot(1,0)
    stg.add_data(sd.yh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)
    p.add_text('Weights no boundary_eval')
    p.reset_camera()

    p.subplot(1,1)
    p.add_text('Weights with boundary_eval')
    stg.add_data(comp_sd.yh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)
    p.reset_camera()
    p.remove_scalar_bar()
    p.write_frame()

    optimizer.zero_grad()
    comp_optimizer.zero_grad()

    loss.backward()
    comp_loss.backward()

    optimizer.step()
    comp_optimizer.step()

p.close()