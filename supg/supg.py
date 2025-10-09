import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from dolfinx import mesh as msh
from supg.param_limit import param_limit
from supg.param_heuristic import param_heuristic
from supg.lagrange_deg import lagrange_deg

class data:
    def __init__(self, domain, Wh, eps, b, c, f, bcs, boundary_eval=True):
        #data for FEM-solution-space
        self.domain = domain

        self.Wh = Wh
        self.u = ufl.TrialFunction(Wh)
        self.v = ufl.TestFunction(Wh)
        self.w = ufl.Coefficient(Wh)
        self.p_limit = param_limit(Wh, eps, b, c)

        #FEM space for the SUPG-parameters/weights
        self.Yh = functionspace(domain, ("DG", 0))
        #self.yh = fem.Function(self.Yh)
        self.yh = param_heuristic(self.Wh, eps, b, self.Yh)
        self.y = ufl.TestFunction(self.Yh)
        self.z = ufl.TrialFunction(self.Yh)
        self.Yh_num_loc_dofs = self.yh.x.index_map.size_local

        #SUPG-forms
        a = (eps * ufl.dot(ufl.grad(self.u), ufl.grad(self.v)) + ufl.dot(b, ufl.grad(self.u)) * self.v + c * self.u * self.v) * ufl.dx
        sh = (-eps * ufl.div(ufl.grad(self.u)) + ufl.dot(b, ufl.grad(self.u)) + c * self.u) * (self.yh * ufl.dot(b, ufl.grad(self.v))) * ufl.dx

        L = f * self.v * ufl.dx
        rh = f * self.yh * ufl.dot(b, ufl.grad(self.v)) * ufl.dx

        self.Rh = ufl.replace(a+sh - L-rh, {self.u:self.w})

        self.uh = fem.Function(self.Wh)
        self.prblm = LinearProblem(a=a + sh, L=L+rh, bcs=bcs, u=self.uh)
        
        self.prblm.solve()

        #initially all cells are included in the loss term and a measure that can be altered with the set_cintegration and remove_DBC_evaluation functions
        cids = np.arange(self.domain.topology.index_map(self.domain.topology.dim).size_local, dtype=np.int32)
        marker = np.ones(cids.size, dtype=np.int32)
        cell_tag = msh.meshtags(self.domain, self.domain.topology.dim, cids, marker)
        dx = ufl.Measure("dx", domain = self.domain, subdomain_data=cell_tag, subdomain_id=1)

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
        if boundary_eval:
            pass
        else:
            self.remove_DBC_evaluation()
        
        #the adjoint problem needs homgenous boundary conditions.
        hom_bcs = [fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), bcs[0].dof_indices()[0], self.Wh)]

        #definition of the adjoint problem
        Rh_w = ufl.derivative(form=self.Rh, coefficient=self.w, argument=self.u)
        D_Ih = ufl.replace(ufl.derivative(form=self.loss, coefficient=self.w, argument=self.v), {self.w:self.uh})
        self.psi = fem.Function(self.Wh)
        self.adj_prblm = LinearProblem(a=ufl.adjoint(Rh_w), L=D_Ih, bcs=hom_bcs, u=self.psi)
        self.adj_prblm.solve()

        
        #volume of the cells needs to be factored out for gradient computation
        cvol = fem.assemble_vector(fem.form(self.y * ufl.dx)).array
        vol_fn = fem.Function(self.Yh)
        vol_fn.x.array[:len(cvol)] = 1/cvol
        vol_fn.x.scatter_forward()

        #gradient problem
        self.grd_fn = fem.Function(self.Yh)
        Rh_y = ufl.replace(ufl.derivative(form=self.Rh, coefficient=self.yh, argument=self.y), {self.w:self.uh, self.v:self.psi})
        self.grd_prblm = LinearProblem(a=self.y * vol_fn * self.z * ufl.dx, L=-Rh_y, bcs=[], u=self.grd_fn)
        self.grd_prblm.solve()

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
        marker = np.ones_like(marker_ids, dtype=np.int32)
        cell_tag = msh.meshtags(self.domain, self.domain.topology.dim, marker_ids, marker)
        form = []
        for integral in self.loss.integrals():
            if integral.integral_type() == 'cell':
                form.append(integral.reconstruct(subdomain_data=cell_tag))
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
        marker = self.loss.integrals_by_type('cell')[0].subdomain_data().indices
        for index in marker:
            if np.intersect1d(self.Wh.dofmap.cell_dofs(index), self.prblm.bcs[0].dof_indices()[0]).size > 0:
                marker = marker[marker!=index]
        self.set_cintegration_domain(marker)




