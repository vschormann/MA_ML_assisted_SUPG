from supg import supg
from supg.sp_problems import dat1 as dat
from supg.plotter import fem_plotter_grid
import pyvista as pv
import numpy as np

sd = supg.data(*dat)
stg = fem_plotter_grid(sd.Wh)

errloc = sd.local_loss().x.array
w = 1/(errloc+(1e-8))
prb=w/w.sum()

cids = np.arange(len(prb))
size = 200
filename = 'random_subdomains.mp4'

p = pv.Plotter(notebook=True, shape=(1,2))
p.open_movie(filename, framerate=100)
for i in range(1000):
    p.clear()

    sd.set_cintegration_domain(np.random.choice(a=cids, size=size, replace=False, p=prb))
    p.subplot(0,0)
    p.add_text('local_loss')
    stg.add_data(sd.local_loss())
    p.camera_position = 'xy'
    p.subplot(0,0)
    p.add_text('gradient')
    stg.add_data(sd.constrained_grad())
    p.add_mesh(stg.grid, show_edges=True)
    p.camera_position = 'xy'
    p.write_frame()
p.close()