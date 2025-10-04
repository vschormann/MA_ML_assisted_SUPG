import ufl
from dolfinx import fem
from mpi4py import MPI
from supg.dof_to_cell import dof_to_cell

def reaction_limit(domain, b, c):
    Ch = fem.functionspace(domain, ('DG', 0))

    dtc = dof_to_cell(domain, Ch)
    if isinstance(c, ufl.Constant):
        if c.value == 0:
            return
    c_fun = fem.Function(Ch)
    c_exp = fem.Expression(c, Ch.element.interpolation_points())
    c_fun.interpolate(c_exp)
    c_vec = c_fun.x.array[dtc]

    cbound_fun = fem.Function(Ch)
    exp = c - ufl.div(b)/2
    cbound_exp = fem.Expression(exp, Ch.element.interpolation_points())
    cbound_fun.interpolate(cbound_exp)
    cbound_loc = max(cbound_fun.x.array)
    cbound = MPI.COMM_WORLD.allreduce(cbound_loc, op=MPI.MAX)
    
    return cbound/c_vec