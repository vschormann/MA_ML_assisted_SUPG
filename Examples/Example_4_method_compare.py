from supg.sp_problems import pde_data1 as pde_data
from supg import supg
import pyvista as pv
import ufl
from dolfinx import fem
from mpi4py import MPI
import dolfinx.mesh as msh

domain = msh.create_unit_square(MPI.COMM_WORLD, 10, 10, msh.CellType.quadrilateral)

sd = supg.data(domain=domain, pde_data=pde_data, boundary_eval=False)

stg = sd.create_stage()

domain = sd.domain
Wh = pde_data(domain)[0]
eps = pde_data(domain)[1]
x = ufl.SpatialCoordinate(domain)

ex_exp = x[0]*(1-ufl.exp(-(1-x[0])/eps))* (1 - (((ufl.exp(-(1-x[1])/eps)  + ufl.exp(-(x[1])/eps)))- ufl.exp(-(1)/eps))/(1-ufl.exp(-1/eps)))


exp = fem.Expression(ex_exp, Wh.element.interpolation_points())

u_ex = fem.Function(Wh)
u_ex.interpolate(exp)
p = pv.Plotter(shape=(2,2))

stg.add_data(u_ex)
p.subplot(0,0)
p.add_text('analytical solution')
grid = stg.grid#.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)


sd.set_weights(0)
stg.add_data(sd.uh)
p.subplot(0,1)
p.add_text('Galerkin approximation (weights = 0)')
grid = stg.grid#.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)

sd.set_weights(0.07)
stg.add_data(sd.uh)
p.subplot(1,0)
p.add_text('SUPG approximation with weights = 0.07')
grid = stg.grid#.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)

sd.set_weights(0.3)
stg.add_data(sd.uh)
p.subplot(1,1)
p.add_text('SUPG approximation with weights = 0.3')
grid = stg.grid#.reflect((1,0,0))
p.add_mesh(grid.warp_by_scalar(), show_edges=True)

p.remove_scalar_bar()
p.show()