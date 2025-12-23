from mpi4py import MPI
import dolfinx
import gmsh
def hemker_domain():
    gmsh.initialize()
    center = (-3,-3,0)
    disc_center = (0,0,0)
    radius = 1
    length = 12
    height = 6

    inner_disk = gmsh.model.occ.addDisk(*disc_center, radius, radius)
    channel = gmsh.model.occ.addRectangle(*center, dx=length, dy=height)
    _, map_to_input = gmsh.model.occ.cut(
                [(2, channel)], [(2, inner_disk)]
            )
    gmsh.model.occ.synchronize()

    channel_idx = [idx for (dim, idx) in map_to_input[0] if dim == 2]
    gmsh.model.addPhysicalGroup(2, channel_idx)
    boundary = gmsh.model.getBoundary([(2, e) for e in channel_idx], recursive=False, oriented=False)
    boundary_idx = [idx for (dim, idx) in boundary if dim == 1]

    gmsh.model.addPhysicalGroup(1, boundary_idx)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(3)
    domain, _, _ = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return domain
import pyvista
from supg.sp_problems import pde_data5
from supg import supg
from supg.plotter import fem_plotter_grid
from dolfinx import fem
from dolfinx import mesh as msh
import numpy as np

domain = hemker_domain()
sd = supg.data(domain=domain, pde_data=pde_data5)
stg = fem_plotter_grid(sd.uh.function_space)
stg.add_data(sd.uh)
p = pyvista.Plotter()

p.add_mesh(stg.grid.warp_by_scalar(), show_edges=True)

p.camera_position = 'xy'
p.show()