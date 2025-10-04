from supg.c_inv import c_inv
from supg.elem_size import elem_size
from supg.reaction_limit import reaction_limit
import numpy as np

def param_limit(space, eps, b, c):
    r_limit = reaction_limit(space.mesh, b, c)
    c = c_inv(space)
    epsilon = eps.value
    h_K = elem_size(space.mesh)
    if r_limit is not None:
        return np.minimum(h_K**2/(epsilon*c), r_limit)
    else:
        return h_K**2/(epsilon*c)