from dolfinx import mesh as msh
from mpi4py import MPI
from dolfinx import fem
from dolfinx import default_scalar_type
import ufl
import numpy as np


#Ex 1
def pde_data1(domain):
    Wh = fem.functionspace(domain, ('P', 2))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = msh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(Wh, domain.topology.dim-1, boundary_facets)
    eps = fem.Constant(domain, default_scalar_type(1e-8))
    b = ufl.as_vector((fem.Constant(domain, default_scalar_type(1.0)),fem.Constant(domain, default_scalar_type(0.0))))
    c = fem.Constant(domain, default_scalar_type(0.0))
    f = fem.Constant(domain, default_scalar_type(1.0))
    bcs = [fem.dirichletbc(fem.Constant(domain, default_scalar_type(0.0)), boundary_dofs, Wh)]
    return Wh,eps,b,c,f,bcs,None


#Ex 2
def pde_data2(domain):
    x = ufl.SpatialCoordinate(domain)
    Wh = fem.functionspace(domain, ('P', 2))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = msh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(Wh, domain.topology.dim-1, boundary_facets)
    eps = fem.Constant(domain, default_scalar_type(10**(-3)))
    b = ufl.as_vector((fem.Constant(domain, default_scalar_type(2.0)),fem.Constant(domain, default_scalar_type(3.0))))
    c = fem.Constant(domain, default_scalar_type(1.0))
    expr = x[0]*x[1]**2 - x[1]**2*ufl.exp(2*(x[0]-1)/(eps)) - x[0]*ufl.exp(3*(x[1]-1)/eps) + ufl.exp((2*(x[0]-1)+3*(x[1]-1))/eps)
    f = -eps * ufl.div(ufl.grad(expr)) + ufl.dot(b,ufl.grad(expr)) + c*expr
    u_exact = fem.Expression(expr, Wh.element.interpolation_points())
    uD = fem.Function(Wh)
    uD.interpolate(u_exact)
    bcs = [fem.dirichletbc(uD, boundary_dofs)]
    return Wh,eps,b,c,f,bcs,None

#Ex 3
def pde_data3(domain):
    x = ufl.SpatialCoordinate(domain)
    i = ufl.Index()
    Wh = fem.functionspace(domain, ('P', 2))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = msh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(Wh, domain.topology.dim-1, boundary_facets)
    eps = fem.Constant(domain, default_scalar_type(10**(-8)))
    b = ufl.as_vector((fem.Constant(domain, default_scalar_type(2.0)),fem.Constant(domain, default_scalar_type(3.0))))
    c = fem.Constant(domain, default_scalar_type(0.0))
    expr = 16*x[0]*(1-x[0])*x[1]*(1-x[1])*(1/2+ufl.atan(2*eps**(-1/2)*(0.25**2-(x[0]-0.5)**2-(x[1]-1/2)**2))/ufl.pi)
    f = -eps * expr.dx(i).dx(i) + b[i]*expr.dx(i) + c*expr
    u_exact = fem.Expression(expr, Wh.element.interpolation_points())
    uD = fem.Function(Wh)
    uD.interpolate(u_exact)
    bcs = [fem.dirichletbc(uD, boundary_dofs)]
    return Wh, eps, b,c,f,bcs,None

#Ex 4

def pde_data4(domain):
    x = ufl.SpatialCoordinate(domain)
    Wh = fem.functionspace(domain, ('P', 2))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = msh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(Wh, domain.topology.dim-1, boundary_facets)
    eps = fem.Constant(domain, default_scalar_type(10**(-8)))
    b = ufl.as_vector((fem.Constant(domain, ufl.cos(-ufl.pi/default_scalar_type(3.0))),fem.Constant(domain, ufl.sin(ufl.pi/default_scalar_type(3.0)))))
    c = fem.Constant(domain, default_scalar_type(0.0))
    expr = ufl.conditional(ufl.Or(ufl.eq(x[0],1), ufl.le(x[1],0.7)), 0, 1)
    f = fem.Constant(domain, default_scalar_type(0.0))
    u_exact = fem.Expression(expr, Wh.element.interpolation_points())
    uD = fem.Function(Wh)
    uD.interpolate(u_exact)
    bcs = [fem.dirichletbc(uD, boundary_dofs)]
    return Wh, eps,b,c,f,bcs,None

#Ex 5

def pde_data5(domain):
    Wh = fem.functionspace(domain, ('P', 2))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    DB_dofs1 = fem.locate_dofs_geometrical(Wh, lambda x: np.isclose(x[0], -3))
    DB_dofs2 = fem.locate_dofs_geometrical(Wh, lambda x: np.isclose(x[0]**2+x[1]**2, 1))
    eps = fem.Constant(domain, default_scalar_type(10**(-4)))
    b = ufl.as_vector((fem.Constant(domain, default_scalar_type(1.0)),fem.Constant(domain, default_scalar_type(0.0))))
    c = fem.Constant(domain, default_scalar_type(0.0))
    f = fem.Constant(domain, default_scalar_type(0.0))
    bcs = [fem.dirichletbc(fem.Constant(domain, default_scalar_type(0.0)), DB_dofs1, Wh), fem.dirichletbc(fem.Constant(domain, default_scalar_type(1.0)), DB_dofs2, Wh)]
    g = fem.Constant(domain, default_scalar_type(0.0))
    return Wh, eps,b,c,f,bcs,g