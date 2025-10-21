from supg.exp_to_fun import exp_to_fun
import ufl
from supg.lagrange_deg import lagrange_deg
from dolfinx import fem
import numpy as np

def param_heuristic(Wh, eps, b, space):
    p = lagrange_deg(Wh)
    norm_b = ufl.sqrt(ufl.dot(b,b))
    Ch = fem.functionspace(Wh.mesh, ('DG', 0))
    ch = fem.Function(Ch)
    testf = ufl.TestFunction(Ch)
    form = fem.form(testf * ufl.dx)
    vec = fem.assemble_vector(form).array
    ch.x.array[:] = 2*np.sqrt(vec)
    alpha = norm_b*ch/(2*p*eps)
    Xi = (1/ufl.tanh(alpha)-1/alpha)
    tau_K = ch/(2*p*ufl.sqrt(ufl.dot(b,b)))*Xi
    retfun = exp_to_fun(tau_K, space)
    return retfun
