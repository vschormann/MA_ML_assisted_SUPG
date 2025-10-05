import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

def exp_to_fun(exp, space):
    ts = ufl.TestFunction(space)
    tr = ufl.TrialFunction(space)
    a = tr * ts * ufl.dx
    L = exp * ts * ufl.dx
    retfun = LinearProblem(a, L, bcs=[]).solve()
    return retfun