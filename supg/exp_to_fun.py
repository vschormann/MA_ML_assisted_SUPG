import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

def exp_to_fun(exp, space, petsc_options):
    ts = ufl.TestFunction(space)
    tr = ufl.TrialFunction(space)
    retfun = fem.Function(space)
    a = tr * ts * ufl.dx
    L = exp * ts * ufl.dx
    prblm = LinearProblem(a=a, L=L, bcs=[], u= retfun, petsc_options=petsc_options)
    prblm.solve()
    return retfun