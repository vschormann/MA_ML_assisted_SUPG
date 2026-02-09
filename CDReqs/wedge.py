from dolfinx import mesh as msh
from mpi4py import MPI
from dolfinx import fem
from dolfinx import default_scalar_type
import ufl
import numpy as np
from utils.FEniCSx_solver import FEniCSx_solver


class wedge(FEniCSx_solver):
    def __init__(
        self,
        comm=MPI.COMM_WORLD, 
        mesh=None, 
        nx=16,
        ny=16,
        cell_type=msh.CellType.quadrilateral,
        p=1,
        eps_val=1e-8,
        b1_val=1.0,
        b2_val=0.0,
        c_val=0.0,
        f_val=1.0
    ):
        
        if mesh == None:
            mesh = msh.create_unit_square(comm=comm, nx=nx, ny=ny, cell_type=cell_type)

        Wh = fem.functionspace(mesh, ('P', p))
        uh = fem.Function(Wh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        boundary_facets = msh.exterior_facet_indices(mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(Wh, mesh.topology.dim-1, boundary_facets)
        eps = fem.Constant(mesh, default_scalar_type(eps_val))
        b = ufl.as_vector((fem.Constant(mesh, default_scalar_type(b1_val)),fem.Constant(mesh, default_scalar_type(b2_val))))
        c = fem.Constant(mesh, default_scalar_type(c_val))
        f = fem.Constant(mesh, default_scalar_type(f_val))
        bcs = [fem.dirichletbc(fem.Constant(mesh, default_scalar_type(0.0)), boundary_dofs, Wh)]

        pde_data = mesh,Wh,uh,eps,b,None,f,None,bcs



        cid_lims = mesh.topology.index_map(2).local_range
        marker_ids = np.arange(cid_lims[0], cid_lims[1])

        for index in range(cid_lims[0],cid_lims[1]):
            if np.intersect1d(Wh.dofmap.cell_dofs(index), bcs[0].dof_indices()[0]).size > 0:
                    marker_ids = marker_ids[marker_ids!=index]

        marker = np.ones_like(marker_ids, dtype=np.int32)
        cell_tag = msh.meshtags(mesh, mesh.topology.dim, marker_ids, marker)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tag, subdomain_id=1)

        x = ufl.SpatialCoordinate(mesh)
        ex_exp = x[0]*(1-ufl.exp(-(1-x[0])/eps))* (1 - ((ufl.exp(-(1-x[1])/eps)  + ufl.exp(-(x[1])/eps))- ufl.exp(-1/eps))/(1-ufl.exp(-1/eps)))

        exp = fem.Expression(ex_exp, Wh.element.interpolation_points())

        self.u_ex = fem.Function(Wh)
        self.u_ex.interpolate(exp)

        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) + c * uh - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
        #loss = (uh-u_ex)**2 * ufl.dx
        loss = residual + crosswind_loss
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
