import dolfinx.mesh as msh
from mpi4py import MPI
from dolfinx import fem
import numpy as np


def dof_to_cell(domain, space):
    domain.topology.create_connectivity(2,2)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    dof_to_cell = []
    for cell in range(num_cells):
        cid = np.intersect1d(fem.locate_dofs_topological(space, domain.topology.dim, [cell]), np.arange(num_cells))
        dof_to_cell.append(int(cid))
    return dof_to_cell