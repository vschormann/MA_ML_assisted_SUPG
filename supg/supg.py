import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from dolfinx import mesh as msh
from supg.param_limit import param_limit
from supg.param_heuristic import param_heuristic
from supg.plotter import fem_plotter_grid

class data:
    def __init__(self, domain, pde_data, boundary_eval=True):
        #data for FEM-solution-space
        self.pde_data = pde_data
        self.domain = domain
        Wh,eps,b,c,f,bcs,g = pde_data(domain)
        u = ufl.TrialFunction(Wh)
        self.v = ufl.TestFunction(Wh)
        self.w = ufl.Coefficient(Wh)
        self.p_limit = param_limit(Wh, eps, b, c)
        dx = ufl.Measure("dx", domain = self.domain)

        #FEM space for the SUPG-parameters/weights
        self.Yh = functionspace(domain, ("DG", 0))
        self.yh = param_heuristic(Wh, eps, b, self.Yh)
        self.y = ufl.TestFunction(self.Yh)
        z = ufl.TrialFunction(self.Yh)

        #number of locally owned cells
        self.Yh_num_loc_dofs = self.yh.x.index_map.size_local

        #SUPG-forms
        a = (eps * ufl.dot(ufl.grad(u), ufl.grad(self.v)) + ufl.dot(b, ufl.grad(u)) * self.v + c * u * self.v) * dx
        sh = (-eps * ufl.div(ufl.grad(u)) + ufl.dot(b, ufl.grad(u)) + c * u) * (self.yh * ufl.dot(b, ufl.grad(self.v))) * dx

        L = f * self.v * dx
        if g is not None:
            L += g * self.v * ufl.ds
        rh = f * self.yh * ufl.dot(b, ufl.grad(self.v)) * dx

        self.Rh = ufl.replace(a+sh - L-rh, {u:self.w})

        self.uh = fem.Function(Wh)
        self.prblm = LinearProblem(a=a + sh, L=L+rh, bcs=bcs, u=self.uh)
        
        self.prblm.solve()

        #loss term is composed of a residual and the crosswind term
        residual = (-eps*ufl.div(ufl.grad(self.w)) + ufl.dot(b, ufl.grad(self.w)) + c * self.w - f)**2 * dx
 
        b_perp = ufl.conditional(ufl.eq(ufl.inner(b, b),0), b, -ufl.perp(b)/ufl.sqrt(ufl.inner(b,b)))
        cross = ufl.max_value(ufl.dot(b_perp, ufl.grad(self.w)), -ufl.dot(b_perp, ufl.grad(self.w)))
        crosswind_loss = ufl.conditional(ufl.lt(cross, 1), 1/2*(5*cross**2 - 3*cross**3), ufl.sqrt(cross)) * dx


        # Jump-term over the facets is negligible when using gradient descend and derived methods.
        #n = ufl.FacetNormal(self.domain)
        #alpha_E = ufl.avg(ufl.CellDiameter(self.domain)) / ufl.sqrt(eps)
        #self.facet_loss = ufl.jump(ufl.grad(self.w), b_perp)**2 *  ufl.dS

        self.loss = residual + crosswind_loss
        
        #the adjoint problem needs homgenous boundary conditions.
        hom_bcs = [fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), bcs[0].dof_indices()[0], Wh)]

        #definition of the adjoint problem
        Rh_w = ufl.derivative(form=self.Rh, coefficient=self.w, argument=u)
        D_Ih = ufl.replace(ufl.derivative(form=self.loss, coefficient=self.w, argument=self.v), {self.w:self.uh})
        self.psi = fem.Function(Wh)
        self.adj_prblm = LinearProblem(a=ufl.adjoint(Rh_w), L=D_Ih, bcs=hom_bcs, u=self.psi)
        self.adj_prblm.solve()

        
        #volume of the cells needs to be factored out for gradient computation
        cvol = fem.assemble_vector(fem.form(self.y * dx)).array
        vol_fn = fem.Function(self.Yh)
        vol_fn.x.array[:len(cvol)] = 1/cvol
        vol_fn.x.scatter_forward()

        #gradient problem
        self.grd_fn = fem.Function(self.Yh)
        Rh_y = ufl.replace(ufl.derivative(form=self.Rh, coefficient=self.yh, argument=self.y), {self.w:self.uh, self.v:self.psi})
        self.grd_prblm = LinearProblem(a=self.y * vol_fn * z * dx, L=-Rh_y, bcs=[], u=self.grd_fn)
        self.grd_prblm.solve()

        if boundary_eval:
            pass
        else:
            self.remove_DBC_evaluation()

    def set_weights(self, weights):
        if isinstance(weights, np.ndarray):
            self.yh.x.array[:self.Yh_num_loc_dofs] = np.clip(a=weights[:self.Yh_num_loc_dofs], a_min=0, a_max=self.p_limit)
        else:
            self.yh.x.array[:self.Yh_num_loc_dofs] = np.clip(a=weights, a_min=0, a_max=self.p_limit)
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
    
    def local_loss(self, loss=None):
        if loss == None:
            loss = self.loss

        for integral in loss.integrals():
            integrand = integral.integrand()
            loss = ufl.replace(loss, {integrand: integrand * self.y})
        loss = ufl.replace(loss, {self.w: self.uh})
        vec = fem.assemble_vector(fem.form(loss)).array
        retfun = fem.Function(self.Yh)
        retfun.x.array[:len(vec)] = vec
        retfun.x.scatter_forward()
        return retfun

    def constrained_grad(self):
        self.grd_prblm.solve()
        return self.grd_fn.x.array[:self.Yh_num_loc_dofs]

    def set_cintegration_domain(self, marker_ids):
        if isinstance(marker_ids, msh.MeshTags):
            cell_tag = marker_ids
        else:
            marker = np.ones_like(marker_ids, dtype=np.int32)
            cell_tag = msh.meshtags(self.domain, self.domain.topology.dim, marker_ids, marker)
        form = []
        for integral in self.loss.integrals():
            if integral.integral_type() == 'cell':
                form.append(integral.reconstruct(subdomain_data=cell_tag, subdomain_id=1))
            else:
                form.append(integral)
        self.loss = ufl.form.Form(form)
        
        D_Ih = ufl.replace(ufl.derivative(form=self.loss, coefficient=self.w, argument=self.v), {self.w:self.uh})
        lhs = self.adj_prblm.a
        hom_bcs = self.adj_prblm.bcs
        self.adj_prblm = LinearProblem(a=lhs, L=D_Ih, bcs=hom_bcs, u=self.psi)
        self.adj_prblm.solve()
        
        self.grd_prblm.b.assemble()
        self.grd_prblm.solve()

    def remove_DBC_evaluation(self):
        try:
            marker = self.loss.integrals_by_type('cell')[0].subdomain_data().indices
        except:
            range = self.domain.topology.index_map(2).local_range
            marker = np.arange(range[0], range[1])
        for index in marker:
            if np.intersect1d(self.uh.function_space.dofmap.cell_dofs(index), self.prblm.bcs[0].dof_indices()[0]).size > 0:
                marker = marker[marker!=index]
        self.set_cintegration_domain(marker)

    def alter_triangulation(self, domain, boundary_eval=True, cell_tag=None):
        self.__init__(domain, self.pde_data, boundary_eval)
        if cell_tag is not None:
            self.set_cintegration_domain(cell_tag)
            

    #This function only works for triangular grids as refinement is not implemented in FEniCSx for quadrilateral grids
    def refine_grid(self, loss=None, threshold=90):
        cell_tag = self.loss.integrals_by_type('cell')[0].subdomain_data()
        #parent_cells = cell_tag.indices
        if str(self.domain.ufl_cell())!='triangle':
            print('automatic grid refinement is only implemented for triangles')
            return
        loss = self.local_loss(loss).x.array
        args = np.argwhere(loss>np.percentile(loss, threshold))
        list = np.array([])
        for arg in args:
            list = np.append(list, self.domain.topology.connectivity(2,1).links(arg)[0])
        refined_domain, parent_cells, _ = msh.refine(self.domain, list, option=msh.RefinementOption.parent_cell)
        if cell_tag is not None:
            refined_cell_tag = msh.transfer_meshtag(cell_tag, refined_domain, parent_cell=parent_cells)
            self.alter_triangulation(refined_domain, cell_tag=refined_cell_tag)
        else:
            self.alter_triangulation(refined_domain, cell_tag=None)

    def create_stage(self):
        return fem_plotter_grid(self.uh.function_space)




