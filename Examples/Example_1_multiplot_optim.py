#from __future__ import annotations

import pyvista as pv
from supg.sp_problems import pde_data1 as pde_data
from supg import supg
from dolfinx import mesh as msh
from mpi4py import MPI

domain = msh.create_unit_square(MPI.COMM_WORLD, 16, 16, msh.CellType.triangle)

sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)
sd.set_weights(1e-1)
stg = sd.create_stage()

weights = sd.yh.x.array

learning_rate = 1e-3

filename = 'optim-loop.mp4'

p = pv.Plotter(notebook=True, shape=(2,2))
p.open_movie(filename)

p.subplot(0,0)
stg.add_data(sd.uh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

p.subplot(0,1)
stg.add_data(sd.yh)
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

p.subplot(1,0)
stg.add_data(sd.local_loss())
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

p.subplot(1,1)
stg.add_data(sd.constrained_grad())
p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)



p.remove_scalar_bar()
p.write_frame()

# Update scalars on each frame
for i in range(200):
    p.clear()
    weights -= learning_rate * sd.constrained_grad()
    sd.set_weights(weights)

    p.subplot(0,0)
    p.add_text(f'Iteration: {i}', name='time-label')
    stg.add_data(sd.uh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

    p.subplot(0,1)
    p.add_text('weights')
    stg.add_data(sd.yh)
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

    p.subplot(1,0)
    p.add_text('loss')
    stg.add_data(sd.local_loss())
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

    p.subplot(1,1)
    p.add_text('gradient')
    stg.add_data(sd.constrained_grad())
    p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

    p.remove_scalar_bar()
    p.write_frame() 


p.close()