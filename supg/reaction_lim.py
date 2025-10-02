import ufl
from dolfinx import fem
from mpi4py import MPI

def reaction_lim(domain, b, c):
    Ch = fem.functionspace(domain, ('DG', 0))
    if isinstance(c, ufl.Constant):
        if c.value == 0:
            return
    c_fun = fem.Function(Ch)
    cbound_fun = fem.Function(Ch)
    exp = c - ufl.div(b)/2
    
    c_exp = fem.Expression(c, Ch.element.interpolation_points())

    cbound_exp = fem.Expression(exp, Ch.element.interpolation_points()
                                )
    c_fun.interpolate(c_exp)
    cbound_fun.interpolate(cbound_exp)

    cmax_loc = max(cbound_fun.x.array)
    cbound_loc = max(c_fun.x.array)
    return MPI.COMM_WORLD.allreduce(cmax_loc, op=MPI.MAX)/MPI.COMM_WORLD.allreduce(cbound_loc, op=MPI.MAX)