from dolfinx import mesh as msh, fem, io, default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
from utils.FEniCSx_solver import FEniCSx_solver
import gmsh


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

        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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


class bump(FEniCSx_solver):
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
        b2_val=0.0
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
        x = ufl.SpatialCoordinate(mesh)
        f = ufl.conditional(
            ufl.Or(
                ufl.ge(abs(x[0] - 0.5), 0.25), 
                ufl.ge(abs(x[1] - 0.5), 0.25)
            ),
            0.0,
            -32.0 * (x[0] - 0.5)
        )

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


        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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
        b2_val=3.0
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
        expr = 16*x[0]*(1-x[0])*x[1]*(1-x[1])*(1/2+ufl.atan(2*eps**(-1/2)*(0.25**2-(x[0]-0.5)**2-(x[1]-1/2)**2))/ufl.pi)
        f = -eps * expr.dx(i).dx(i) + b[i]*expr.dx(i)
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


class lifted_edge(FEniCSx_solver):
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
        c_val=1.0
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
        x = ufl.SpatialCoordinate(mesh)
        expr = x[0]*x[1]**2 - x[1]**2*ufl.exp(2*(x[0]-1)/(eps)) - x[0]*ufl.exp(3*(x[1]-1)/eps) + ufl.exp((2*(x[0]-1)+3*(x[1]-1))/eps)
        f = -eps * ufl.div(ufl.grad(expr)) + ufl.dot(b,ufl.grad(expr)) + c*expr
        u_exact = fem.Expression(expr, Wh.element.interpolation_points())
        uD = fem.Function(Wh)
        uD.interpolate(u_exact)
        bcs = [fem.dirichletbc(uD, boundary_dofs)]

        pde_data = mesh,Wh,uh,eps,b,c,f,None,bcs



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

        self.uD = fem.Function(Wh)
        self.uD.interpolate(exp)

        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) + c * uh - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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


class falloff(FEniCSx_solver):
    def __init__(
        self,
        comm=MPI.COMM_WORLD, 
        mesh=None, 
        nx=16,
        ny=16,
        cell_type=msh.CellType.quadrilateral,
        p=1,
        eps_val=1e-8,
        b1_val=3.0,
        b2_val=3.0,
        f_val=0.0
    ):
        
        if mesh == None:
            mesh = msh.create_unit_square(comm=comm, nx=nx, ny=ny, cell_type=cell_type)

        Wh = fem.functionspace(mesh, ('P', p))
        uh = fem.Function(Wh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        boundary_facets = msh.exterior_facet_indices(mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(Wh, mesh.topology.dim-1, boundary_facets)


        eps = fem.Constant(mesh, default_scalar_type(eps_val))
        b = ufl.as_vector((fem.Constant(mesh, ufl.cos(-ufl.pi/default_scalar_type(b1_val))),fem.Constant(mesh, ufl.sin(ufl.pi/default_scalar_type(b2_val)))))
        f = fem.Constant(mesh, default_scalar_type(f_val))
        x = ufl.SpatialCoordinate(mesh)
        expr = ufl.conditional(ufl.Or(ufl.eq(x[0],1), ufl.le(x[1],0.7)), 0, 1)
        u_exact = fem.Expression(expr, Wh.element.interpolation_points())
        uD = fem.Function(Wh)
        uD.interpolate(u_exact)
        bcs = [fem.dirichletbc(uD, boundary_dofs)]

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

        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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


class hemker(FEniCSx_solver):
    def __init__(
        self,
        comm=MPI.COMM_WORLD, 
        mesh=None, 
        cell_type=msh.CellType.quadrilateral,
        p=1,
        eps_val=1e-4,
        b1_val=1.0,
        b2_val=0.0,
        f_val=0.0,
        g_val=0
    ):
        
        if mesh == None:
            gmsh.initialize()
            center = (-3,-3,0)
            disc_center = (0,0,0)
            radius = 1
            length = 12
            height = 6

            inner_disk = gmsh.model.occ.addDisk(*disc_center, radius, radius)
            channel = gmsh.model.occ.addRectangle(*center, dx=length, dy=height)
            _, map_to_input = gmsh.model.occ.cut(
                        [(2, channel)], [(2, inner_disk)]
                    )
            gmsh.model.occ.synchronize()

            channel_idx = [idx for (dim, idx) in map_to_input[0] if dim == 2]
            gmsh.model.addPhysicalGroup(2, channel_idx)
            boundary = gmsh.model.getBoundary([(2, e) for e in channel_idx], recursive=False, oriented=False)
            boundary_idx = [idx for (dim, idx) in boundary if dim == 1]

            gmsh.model.addPhysicalGroup(1, boundary_idx)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(1)
            mesh, _, _ = io.gmshio.model_to_mesh(
                gmsh.model, comm, 0, gdim=2)
            gmsh.finalize()

        
    

        Wh = fem.functionspace(mesh, ('P', p))
        uh = fem.Function(Wh)
        

        eps = fem.Constant(mesh, default_scalar_type(eps_val))
        b = ufl.as_vector((fem.Constant(mesh, default_scalar_type(b1_val)),fem.Constant(mesh, default_scalar_type(b2_val))))
        f = fem.Constant(mesh, default_scalar_type(f_val))
        g = fem.Constant(mesh, default_scalar_type(g_val))
        x = ufl.SpatialCoordinate(mesh)
        expr = ufl.conditional(ufl.Or(ufl.eq(x[0],1), ufl.le(x[1],0.7)), 0, 1)
        u_exact = fem.Expression(expr, Wh.element.interpolation_points())
        uD = fem.Function(Wh)
        uD.interpolate(u_exact)


        DB_dofs1 = fem.locate_dofs_geometrical(Wh, lambda x: np.isclose(x[0], -3))
        DB_dofs2 = fem.locate_dofs_geometrical(Wh, lambda x: np.isclose(x[0]**2+x[1]**2, 1))
        bcs = [fem.dirichletbc(fem.Constant(mesh, default_scalar_type(0.0)), DB_dofs1, Wh), fem.dirichletbc(fem.Constant(mesh, default_scalar_type(1.0)), DB_dofs2, Wh)]
        
        pde_data = mesh,Wh,uh,eps,b,None,f,g,bcs



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

        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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

    
class curved_wall(FEniCSx_solver):
    def __init__(
        self,
        comm=MPI.COMM_WORLD, 
        mesh=None, 
        nx=16,
        ny=16,
        cell_type=msh.CellType.quadrilateral,
        p=1,
        eps_val=1e-8,
        f_val = 0.0
    ):
        
        if mesh == None:
            mesh = msh.create_unit_square(comm=comm, nx=nx, ny=ny, cell_type=cell_type)

        Wh = fem.functionspace(mesh, ('P', p))
        uh = fem.Function(Wh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        boundary_facets = msh.exterior_facet_indices(mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(Wh, mesh.topology.dim-1, boundary_facets)
        eps = fem.Constant(mesh, default_scalar_type(eps_val))
        x = ufl.SpatialCoordinate(mesh)
        b = ufl.as_vector((-x[1], x[0]))
        f = fem.Constant(mesh, default_scalar_type(f_val))

        expr = ufl.conditional(ufl.And(ufl.And(ufl.ge(x[0],1/3), ufl.le(x[0],2/3)), ufl.eq(x[1],0)), 1, 0)
        u_exact = fem.Expression(expr, Wh.element.interpolation_points())
        uD = fem.Function(Wh)
        uD.interpolate(u_exact)
        bcs = [fem.dirichletbc(uD, boundary_dofs)]

        pde_data = mesh,Wh,uh,eps,b,None,f,None,bcs



        cid_lims = mesh.topology.index_map(2).local_range
        marker_ids = np.arange(cid_lims[0], cid_lims[1])

        for index in range(cid_lims[0],cid_lims[1]):
            if np.intersect1d(Wh.dofmap.cell_dofs(index), bcs[0].dof_indices()[0]).size > 0:
                    marker_ids = marker_ids[marker_ids!=index]

        marker = np.ones_like(marker_ids, dtype=np.int32)
        cell_tag = msh.meshtags(mesh, mesh.topology.dim, marker_ids, marker)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tag, subdomain_id=1)


        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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


class curved_wave(FEniCSx_solver):
    def __init__(
        self,
        comm=MPI.COMM_WORLD, 
        mesh=None, 
        nx=16,
        ny=16,
        cell_type=msh.CellType.quadrilateral,
        p=1,
        eps_val=1e-8,
        f_val = 0.0,
        g_val = 0.0
    ):
        
        if mesh == None:
            mesh = msh.create_unit_square(comm=comm, nx=nx, ny=ny, cell_type=cell_type)

        Wh = fem.functionspace(mesh, ('P', p))
        uh = fem.Function(Wh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        eps = fem.Constant(mesh, default_scalar_type(eps_val))
        x = ufl.SpatialCoordinate(mesh)
        b = ufl.as_vector((-x[1], x[0]))
        f = fem.Constant(mesh, default_scalar_type(f_val))
        g = fem.Constant(mesh, default_scalar_type(g_val))

        DB_dofs1 = fem.locate_dofs_geometrical(Wh, lambda x: np.isclose(x[1],0))
        DB_dofs2 = fem.locate_dofs_geometrical(Wh, lambda x: (np.isclose(x[1],1)|np.isclose(x[0],1)))
        expr = ufl.conditional(
            ufl.le(x[0],1/3), 
            x[0], 
            ufl.conditional(
                ufl.And(
                    ufl.gt(x[0], 1/3), 
                    ufl.lt(x[0], 2/3)
                ),
                1/3+x[0], 
                1-x[0]
            )
        )
        u_exact = fem.Expression(expr, Wh.element.interpolation_points())
        uD = fem.Function(Wh)
        uD.interpolate(u_exact)
        bcs = [fem.dirichletbc(uD, DB_dofs1), fem.dirichletbc(fem.Constant(mesh, default_scalar_type(0.0)), DB_dofs2, Wh)]

        pde_data = mesh,Wh,uh,eps,b,None,f,g,bcs



        cid_lims = mesh.topology.index_map(2).local_range
        marker_ids = np.arange(cid_lims[0], cid_lims[1])

        for index in range(cid_lims[0],cid_lims[1]):
            if np.intersect1d(Wh.dofmap.cell_dofs(index), bcs[0].dof_indices()[0]).size > 0:
                    marker_ids = marker_ids[marker_ids!=index]

        marker = np.ones_like(marker_ids, dtype=np.int32)
        cell_tag = msh.meshtags(mesh, mesh.topology.dim, marker_ids, marker)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tag, subdomain_id=1)


        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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


class curved_waves(FEniCSx_solver):
    def __init__(
        self,
        comm=MPI.COMM_WORLD, 
        mesh=None, 
        nx=16,
        ny=16,
        cell_type=msh.CellType.quadrilateral,
        p=1,
        eps_val=1e-8,
        f_val = 0.0,
        g_val = 0.0
    ):
        
        if mesh == None:
            mesh = msh.create_unit_square(comm=comm, nx=nx, ny=ny, cell_type=cell_type)

        Wh = fem.functionspace(mesh, ('P', p))
        uh = fem.Function(Wh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        eps = fem.Constant(mesh, default_scalar_type(eps_val))
        x = ufl.SpatialCoordinate(mesh)
        b = ufl.as_vector((-x[1], x[0]))
        f = fem.Constant(mesh, default_scalar_type(f_val))
        g = fem.Constant(mesh, default_scalar_type(g_val))

        DB_dofs1 = fem.locate_dofs_geometrical(Wh, lambda x: np.isclose(x[1],0))
        DB_dofs2 = fem.locate_dofs_geometrical(Wh, lambda x: (np.isclose(x[1],1)|np.isclose(x[0],1)))
        expr = ufl.conditional(
            ufl.le(x[0],1/6), 
            6*x[0], 
            ufl.conditional(
                ufl.And(
                    ufl.ge(x[0], 1/6), 
                    ufl.le(x[0], 1/3)
                ),
                2-6*x[0],
                ufl.conditional(
                    ufl.And(
                        ufl.ge(x[0], 1/3), 
                        ufl.le(x[0], 1/2)
                    ),
                    -2+6*x[0],
                    ufl.conditional(
                        ufl.And(
                            ufl.ge(x[0], 1/2), 
                            ufl.le(x[0], 2/3)
                        ),
                        4-6*x[0],
                        ufl.conditional(
                            ufl.And(
                                ufl.ge(x[0], 2/3), 
                                ufl.le(x[0], 5/6)
                            ),
                            -4+6*x[0],
                            6-6*x[0]
                        )
                    )
                )
            )
        )
        u_exact = fem.Expression(expr, Wh.element.interpolation_points())
        uD = fem.Function(Wh)
        uD.interpolate(u_exact)
        bcs = [fem.dirichletbc(uD, DB_dofs1), fem.dirichletbc(fem.Constant(mesh, default_scalar_type(0.0)), DB_dofs2, Wh)]

        pde_data = mesh,Wh,uh,eps,b,None,f,g,bcs



        cid_lims = mesh.topology.index_map(2).local_range
        marker_ids = np.arange(cid_lims[0], cid_lims[1])

        for index in range(cid_lims[0],cid_lims[1]):
            if np.intersect1d(Wh.dofmap.cell_dofs(index), bcs[0].dof_indices()[0]).size > 0:
                    marker_ids = marker_ids[marker_ids!=index]

        marker = np.ones_like(marker_ids, dtype=np.int32)
        cell_tag = msh.meshtags(mesh, mesh.topology.dim, marker_ids, marker)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tag, subdomain_id=1)


        residual = (-eps*ufl.div(ufl.grad(uh)) + ufl.dot(b, ufl.grad(uh)) - f)**2 * dx

        b_perp = ufl.as_vector((fem.Constant(mesh, default_scalar_type(0.0)),fem.Constant(mesh, default_scalar_type(-1.0))))
        cross = abs(ufl.dot(b_perp, ufl.grad(uh)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx
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