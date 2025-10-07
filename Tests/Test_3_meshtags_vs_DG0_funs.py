from dolfinx import fem
from dolfinx import mesh
from mpi4py import MPI
import ufl
import numpy as np
import time

iterations = 1000
size = 500
domain = mesh.create_unit_square(MPI.COMM_WORLD, 500, 500, mesh.CellType.quadrilateral)
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

Yh = fem.functionspace(domain, ('DG', 0))
yh = fem.Function(Yh)

yh.x.array[:] = np.arange(len(yh.x.array))

yh.x.scatter_forward()

all_cells = np.arange(len(yh.x.array), dtype=np.int32)


marker = np.zeros(all_cells.size, dtype=np.int32)
batch = np.random.choice(all_cells, size=size, replace=False)
marker[batch] = 1
cell_tag = mesh.meshtags(domain, domain.topology.dim, np.arange(all_cells.size, dtype=np.int32), marker)
dx = ufl.Measure("dx", domain = domain, subdomain_data = cell_tag, subdomain_id = 1)

exp = yh*dx
start = time.perf_counter()
for i in range(iterations):
    batch = np.random.randint(low=0, high=all_cells.size, size=size, dtype=np.int32)
    marker = np.ones_like(batch)
    cell_tag = mesh.meshtags(domain, domain.topology.dim, batch, marker)
    form = []
    for integral in exp.integrals_by_type('cell'):
        form.append(integral.reconstruct(subdomain_data=cell_tag))
    form = ufl.form.Form(form)
    fem.assemble_scalar(fem.form(form))
end = time.perf_counter()
print(f"Execution time with meshtags: {end - start:.6f} seconds")


mfun = fem.Function(Yh)

dx = ufl.Measure("dx", domain = domain)
exp = yh*mfun*dx
start = time.perf_counter()
for i in range(iterations):
    marker = np.zeros(all_cells.size, dtype=np.int32)
    batch = np.random.choice(all_cells, size=size, replace=False)
    marker[batch] = 1
    mfun.x.array[:] = marker
    mfun.x.scatter_forward()
    fem.assemble_scalar(fem.form(exp))
end = time.perf_counter()
print(f"Execution time with DG0 function: {end - start:.6f} seconds")