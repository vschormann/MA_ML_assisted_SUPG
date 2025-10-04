from dolfinx import mesh as msh
from dolfinx import fem
import ufl
import numpy as np 

class facet_eval():
        def __init__(self, domain, exp, w):
            self.w = w

            # Make sure connectivity exists
            domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)  # facets -> cells

            # Get facets-to-cells connectivity
            facet_to_cell = domain.topology.connectivity(domain.topology.dim - 1, domain.topology.dim)

            # Interior facets have exactly 2 connected cells
            interior_facets = [f for f in range(domain.topology.index_map(domain.topology.dim - 1).size_global) 
                               if len(facet_to_cell.links(f)) == 2]
            
            self.int_facet_mesh, _, _, _ = msh.create_submesh(domain, 1, np.array(interior_facets))

            self.int_facet_mesh.topology.create_connectivity(1,1)

            self.projecting_space = fem.functionspace(domain, ('CG', 1))

            self.parent_dofs = fem.locate_dofs_topological(self.projecting_space, domain.topology.dim - 1, interior_facets)

            self.projection_canvas = fem.functionspace(self.int_facet_mesh, ('CG', 1))

            #self.pc = ufl.TestFunction(self.projection_canvas)
            self.projection = fem.Function(self.projection_canvas)
            self.exp = ufl.form.Form([I for I in exp.integrals() if I.integral_type() == 'interior_facet'])
            for int_fct_integral in self.exp.integrals():
                integrand = int_fct_integral.integrand()
                self.exp = ufl.replace(self.exp, {integrand: integrand * ufl.TestFunction(self.projecting_space)('+') })


            self.Fh = fem.functionspace(self.int_facet_mesh, ('DG', 0))
            self.f = ufl.TestFunction(self.Fh)
            self.facet_jump = fem.Function(self.Fh)

            self.vol = fem.assemble_vector(fem.form((ufl.TestFunction(self.projection_canvas) * ufl.dx))).array

            self.fcts = np.arange(self.int_facet_mesh.topology.index_map(self.int_facet_mesh.topology.dim).size_global)
        
        def compute_facet_loss(self, uh):
            frm = ufl.replace(self.exp, {self.w: uh})

            v = fem.assemble_vector(fem.form(frm)).array

            self.projection.x.array[:] = v[self.parent_dofs]/self.vol
            self.projection.x.scatter_forward()

            
            for f in self.fcts:
                self.facet_jump.x.array[fem.locate_dofs_topological(self.Fh, 1, [f])] = np.average(self.projection.x.array[fem.locate_dofs_topological(self.projection_canvas, 1, [f])])
                self.facet_jump.x.scatter_forward()

            self.facet_jump.x.array[:] = fem.assemble_vector(fem.form(self.facet_jump * self.f * ufl.dx)).array
            self.facet_jump.x.scatter_forward()