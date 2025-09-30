import numpy as np

def lagrange_deg(space):
    shape = space.mesh.topology.cell_name()
    dim = space.element.space_dimension
    if shape == 'triangle':
        deg = ((8*dim + 1)**0.5-3)/2
    if shape =='quadrilateral':
        deg = (dim)**0.5-1
    return int(deg)

