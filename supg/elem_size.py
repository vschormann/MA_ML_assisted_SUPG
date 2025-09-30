from supg.dof_to_cell import dof_to_cell
import numpy as np
from dolfinx import fem


def elem_size(domain):
    Ch = fem.functionspace(domain, ('DG', 0))
    dtc = dof_to_cell(domain, Ch)
    h_K = np.array([])
    if domain.topology.cell_name() == 'quadrilateral':
        for cell in domain.geometry.x[domain.geometry.dofmap]:
            coords = cell[:,:domain.topology.dim]
            hgt = np.linalg.norm(coords[0]-coords[1])
            wdt = np.linalg.norm(coords[0]-coords[2])
            elem_size = hgt*wdt*np.sqrt(2/(hgt**2 + wdt**2))
            h_K = np.append(h_K, elem_size)
    if domain.topology.cell_name() == 'triangle':
        for cell in domain.geometry.x[domain.geometry.dofmap]:
            coords = cell[:,:domain.topology.dim]
            a = np.linalg.norm(coords[0]-coords[1])
            b = np.linalg.norm(coords[0]-coords[2])
            c = np.linalg.norm(coords[1]-coords[2])
            s = (a+b+c)/2
            A = np.sqrt(s*((s-a)*(s-b)*(s-c)))
            elem_size = 4*A/np.sqrt(a**2+b**2+c**2)
            h_K = np.append(h_K, elem_size)
    return h_K[dtc]