import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from dolfinx import mesh as msh

class data:
    def __init__(self, domain, Wh, eps, b, c, f, bcs, boundary_eval=True):
        #data for FEM-solution-space
        self.domain = domain

        self.Wh = Wh
        self.u = ufl.TrialFunction(Wh)
        self.v = ufl.TestFunction(Wh)
        self.w = ufl.Coefficient(Wh)
        
        #problem data
        self.eps = eps
        self.b = b
        self.c = c
        self.f = f
        self.bcs = bcs

        #FEM space for the SUPG-parameters/weights
        self.Yh = functionspace(domain, ("DG", 0))
        self.yh = fem.Function(self.Yh)
        self.y = ufl.TestFunction(self.Yh)
        self.z = ufl.TrialFunction(self.Yh)

        #SUPG-forms
        a = (eps * ufl.dot(ufl.grad(self.u), ufl.grad(self.v)) + ufl.dot(b, ufl.grad(self.u)) * self.v + c * self.u * self.v) * ufl.dx
        sh = (-eps * ufl.div(ufl.grad(self.u)) + ufl.dot(b, ufl.grad(self.u)) + c * self.u) * (self.yh * ufl.dot(b, ufl.grad(self.v))) * ufl.dx

        L = f * self.v * ufl.dx
        rh = f * self.yh * ufl.dot(b, ufl.grad(self.v)) * ufl.dx

        Rh = ufl.replace(a+sh - L-rh, {self.u:self.w})

        self.uh = fem.Function(self.Wh)
        self.prblm = LinearProblem(a=a + sh, L=L+rh, bcs=bcs, u=self.uh)
        
        self.prblm.solve()

        #creates meshtags object and measure that ignores the boundary cells if boundary_eval is set to True. Can/should be changed to custom marker later?
        all_cells = np.arange(self.yh.x.array.size, dtype=np.int32)

        self.limit = 1/(24*all_cells.size)

        self.marker = np.ones(all_cells.size, dtype=np.int32)

        if boundary_eval:
            self.dx = ufl.Measure("dx", domain = domain)
        else:
            for index in all_cells:
                if np.intersect1d(Wh.dofmap.cell_dofs(index), bcs[0].dof_indices()[0]).size == 0:
                    self.marker[index] = 0
            cell_tag = msh.meshtags(domain, domain.topology.dim, np.arange(all_cells.size, dtype=np.int32), self.marker)
            self.dx = ufl.Measure("dx", domain = domain, subdomain_data = cell_tag, subdomain_id = 0)

        self.residual = (-self.eps*ufl.div(ufl.grad(self.w)) + ufl.dot(self.b, ufl.grad(self.w)) + self.c * self.w - self.f)**2 * self.dx

 
        b_perp = ufl.conditional(ufl.eq(ufl.inner(self.b,self.b),0), self.b, -ufl.perp(self.b)/ufl.sqrt(ufl.inner(self.b,self.b)))

        cross = ufl.max_value(ufl.dot(b_perp, ufl.grad(self.w)), -ufl.dot(b_perp, ufl.grad(self.w)))
        self.crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * self.dx


        # Jump-term over the facets is negligible when using gradient descend and derived methods.
        #n = ufl.FacetNormal(self.domain)
        #alpha_E = ufl.avg(ufl.CellDiameter(self.domain)) / ufl.sqrt(self.eps)
        self.facet_loss = ufl.jump(ufl.grad(self.w), b_perp)**2 *  ufl.dS

        self.loss = self.residual + self.crosswind_loss

        hom_bcs = [fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), bcs[0].dof_indices()[0], self.Wh)]
        cvol = fem.assemble_vector(fem.form(self.y * ufl.dx)).array
        vol_fn = fem.Function(self.Yh)
        vol_fn.x.array[:] = 1/cvol
        vol_fn.x.scatter_forward()

        Rh_w = ufl.derivative(form=Rh, coefficient=self.w, argument=self.u)
        D_Ih = ufl.replace(ufl.derivative(form=self.loss, coefficient=self.w, argument=self.v), {self.w:self.uh})
        self.psi = fem.Function(self.Wh)
        self.adj_prblm = LinearProblem(a=ufl.adjoint(Rh_w), L=D_Ih, bcs=hom_bcs, u=self.psi, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.adj_prblm.solve()

        self.grd_fn = fem.Function(self.Yh)
        Rh_y = ufl.replace(ufl.derivative(form=Rh, coefficient=self.yh, argument=self.y), {self.w:self.uh, self.v:self.psi})
        self.grd_prblm = LinearProblem(a=self.y * vol_fn * self.z * ufl.dx, L=-Rh_y, bcs=[], u=self.grd_fn, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.grd_prblm.solve()

    def set_weights(self, weights):
        self.yh.x.array[:] = np.clip(weights, 0,1)
        self.yh.x.scatter_forward()
        self.prblm.A.assemble()
        self.prblm.b.assemble()
        self.prblm.solve()

        self.adj_prblm.A.assemble()
        self.adj_prblm.b.assemble()
        self.adj_prblm.solve()

        self.grd_prblm.b.assemble()
        self.grd_prblm.solve()
    
    def global_loss(self, loss=None):
        if loss == None:
            loss = self.loss
        frm = ufl.replace(loss, {self.w:self.uh})
        return fem.assemble_scalar(fem.form(frm))
    
    def local_loss(self, loss=None, itype='cell'):
        if loss == None:
            loss = self.loss
        loss = ufl.form.Form([I for I in loss.integrals() if I.integral_type() == itype])
        match itype:
            case 'cell':
                for cell_integral in loss.integrals():
                    integrand = cell_integral.integrand()
                    loss = ufl.replace(loss, {integrand: integrand * self.y})
                loss = ufl.replace(loss, {self.w: self.uh})
                vec = fem.assemble_vector(fem.form(loss)).array
                retfun = fem.Function(self.Yh)
                retfun.x.array[:] = vec
                retfun.x.scatter_forward()
                return retfun
            case 'interior_facet':
                try:
                    self.fe.compute_facet_loss(self.uh)
                except:
                    self.include_local_facet_loss(loss)
                self.fe.compute_facet_loss(self.uh)
                return self.fe.facet_jump


    def constrained_grad(self):
        
        return self.grd_fn.x.array



    def include_local_facet_loss(self):
        try:
            self.fe = facet_eval(self.domain, self.facet_loss, self.w)
        except:
            self.include_facet_loss()
            self.fe = facet_eval(self.domain, self.facet_loss, self.w)

        self.update_facet_loss()

    def update_facet_loss(self):
        self.fe.compute_facet_loss(self.uh)

    def disregard_local_facet_loss(self):
        del self.fe



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



