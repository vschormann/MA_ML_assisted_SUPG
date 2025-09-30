from supg.lagrange_deg import lagrange_deg
def c_inv(space):
    shape = space.mesh.topology.cell_name()
    degree = lagrange_deg(space)
    if shape == 'triangle':
        match(degree):
            case 1:
                return 0
            case 2:
                return 48
            case 3:
                return 149.1
    if shape == 'quadrilateral':
        match(degree):
            case 1:
                return 0
            case 2:
                return 24
            case 3:
                return 113.2
                
            
