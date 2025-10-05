from supg.exp_to_fun import exp_to_fun
import ufl
from supg.lagrange_deg import lagrange_deg

def param_heuristic(Wh, eps, b, space):
    p = lagrange_deg(Wh)
    norm_b = ufl.sqrt(ufl.dot(b,b))
    alpha = norm_b*ufl.CellDiameter(Wh.mesh)/(2*p*eps)
    Xi = (1/ufl.tanh(alpha)-1/alpha)
    tau_K = ufl.CellDiameter(Wh.mesh)/(2*p*ufl.sqrt(ufl.dot(b,b)))*Xi
    return exp_to_fun(tau_K, space)