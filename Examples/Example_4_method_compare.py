from supg.sp_problems import dat1, stg
from supg import supg
import pyvista as pv
import ufl
from dolfinx import fem

sd = supg.data(*dat1)

domain = sd.domain
eps = sd.eps
Wh = sd.Wh
x = ufl.SpatialCoordinate(sd.domain)

ex_exp = x[0]*(1-ufl.exp(-(1-x[0])/eps))* (1 - (((ufl.exp(-(1-x[1])/eps)  + ufl.exp(-(x[1])/eps)))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))


exp = fem.Expression(ex_exp, Wh.element.interpolation_points())

u_ex = fem.Function(Wh)
u_ex.interpolate(exp)
p = pv.Plotter(shape=(2,2))

stg.add_data(u_ex)
p.subplot(0,0)
p.add_text('analytic solution')
grid = stg.grid.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)


stg.add_data(sd.uh)
sd.set_weights(0)
p.subplot(0,1)
p.add_text('Galerkin approximation (weights = 0)')
grid = stg.grid.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)

sd.set_weights(0.025)
p.subplot(1,0)
p.add_text('SUPG approximation with weights = 0.025')
grid = stg.grid.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)

sd.set_weights(0.1)
p.subplot(1,1)
p.add_text('SUPG approximation with weights = 0.1')
grid = stg.grid.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)

p.remove_scalar_bar()
p.show()