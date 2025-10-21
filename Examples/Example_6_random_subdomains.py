from supg import supg
from supg.sp_problems import pde_data1 as pde_data
from supg.plotter import fem_plotter_grid
import pyvista as pv
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as msh
import ufl

domain = msh.create_unit_square(MPI.COMM_WORLD, 16, 16, msh.CellType.triangle)
sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=True)
stg = sd.create_stage()

errloc = sd.local_loss().x.array
w = 1/(errloc+(1e-8))
prb=w/w.sum()

cids = np.arange(len(prb))
size = int(len(sd.yh.x.array)*0.7)
filename = 'random_subdomains.mp4'

p = pv.Plotter(notebook=True, shape=(1,2))
p.open_movie(filename, framerate=5)
for i in range(100):
    sd.set_cintegration_domain(np.random.choice(a=cids, size=size, replace=False, p=prb))

    p.clear()
    p.subplot(0,0)
    p.add_text('gradient')
    stg.add_data(sd.local_loss())

    p.add_mesh(stg.grid, show_edges=True)
    p.camera_position = 'xy'

    p.subplot(0,1)
    p.add_text('domain')
    exp = sd.loss
    integral = exp.integrals_by_type('cell')[0]
    frm = ufl.form.Form([ufl.replace(integral, {integral.integrand(): 5})])
    vec = sd.local_loss(loss=frm)

    stg.add_data(vec)
    p.add_mesh(stg.grid, show_edges=True)
    p.camera_position = 'xy'
    p.remove_scalar_bar()
    p.write_frame()
p.close()