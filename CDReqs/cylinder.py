from utils.FEniCSx_solver import FEniCSx_solver
from dolfinx import mesh as msh, fem, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
from utils.FEniCSx_solver import FEniCSx_solver


class cylinder(FEniCSx_solver):
    def __init__(
        self,
        comm=MPI.COMM_WORLD, 
        mesh=None, 
        nx=16,
        ny=16,
        cell_type=msh.CellType.quadrilateral,
        p=1,
        eps_val=1e-3,
        b1_val=2.0,
        b2_val=3.0,
        c_val=0.0
        ):
        if mesh == None:
            mesh = msh.create_unit_square(comm=comm, nx=nx, ny=ny, cell_type=cell_type)

        x = ufl.SpatialCoordinate(mesh)
        i = ufl.Index()
        Wh = fem.functionspace(mesh, ('P', p))
        uh = fem.Function(Wh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        boundary_facets = msh.exterior_facet_indices(mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(Wh, mesh.topology.dim-1, boundary_facets)
        eps = fem.Constant(mesh, default_scalar_type(eps_val))
        b = ufl.as_vector((fem.Constant(mesh, default_scalar_type(b1_val)),fem.Constant(mesh, default_scalar_type(b2_val))))
        c = fem.Constant(mesh, default_scalar_type(c_val))
        expr = 16*x[0]*(1-x[0])*x[1]*(1-x[1])*(1/2+ufl.atan(2*eps**(-1/2)*(0.25**2-(x[0]-0.5)**2-(x[1]-1/2)**2))/ufl.pi)
        f = -eps * expr.dx(i).dx(i) + b[i]*expr.dx(i) + c*expr
        u_exact = fem.Expression(expr, Wh.element.interpolation_points())
        uD = fem.Function(Wh)

        uD.interpolate(u_exact)

        bcs = [fem.dirichletbc(uD, boundary_dofs)]

        pde_data = mesh,Wh,uh,eps,b,None,f,None,bcs

        loss = (uh-uD)**2 * ufl.dx
        super().__init__(pde_data=pde_data, loss_form=loss)

        norm_b = ufl.sqrt(ufl.dot(b,b))
        h = ufl.CellDiameter(domain=mesh) 
        alpha = norm_b*h/(2*eps)
        Xi = (1/ufl.tanh(alpha)-1/alpha)
        tau_K = h/(2*norm_b)*Xi
        Th = fem.functionspace(mesh, ('DG', 0))
        tau = fem.Function(Th)
        tau_exp = fem.Expression(tau_K, Th.element.interpolation_points())
        tau.interpolate(tau_exp)
        self.set_weights(tau.x.array)


        self.upper = 100*self.yh.x.array