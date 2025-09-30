from supg.c_inv import c_inv
from supg.elem_size import elem_size

def param_limit(space, eps, reaction_lim=None):
    c = c_inv(space)
    h_K = elem_size(space.mesh)
    if reaction_lim is not None:
        return min(h_K**2/(eps*c), reaction_lim)
    else:
        return h_K**2/(eps*c)