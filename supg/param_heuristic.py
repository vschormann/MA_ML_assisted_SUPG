from supg import exp_to_fun

def param_heuristic(h_K, eps, b, norm, xi, space):
    peclet = norm(b)*h_K/(2*eps)
    exp = h_K/(2*norm(b))*xi(peclet)
    retfun = exp_to_fun(exp, space)