import ufl
from dolfinx import default_scalar_type, fem, la, mesh as msh
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import scipy


class LinearSolver:
    def __init__(self, a, L, uh, bcs):
        self.a_compiled = fem.form(a)
        self.L_compiled = fem.form(L)
        self.A = fem.create_matrix(self.a_compiled)
        self.b = fem.Function(uh.function_space)
        self.bcs = bcs
        self._A_scipy = self.A.to_scipy()
        self.uh = uh

    def solve(self):
        self._A_scipy.data[:] = 0

        fem.assemble_matrix(self.A, self.a_compiled, bcs=self.bcs)

        self.b.x.array[:] = 0
        fem.assemble_vector(self.b.x.array, self.L_compiled)
        fem.apply_lifting(self.b.x.array, [self.a_compiled], [self.bcs])
        self.b.x.scatter_reverse(la.InsertMode.add)
        [bc.set(self.b.x.array) for bc in self.bcs]

        A_inv = scipy.sparse.linalg.splu(self._A_scipy)
        self.uh.x.array[:] = A_inv.solve(self.b.x.array)
        return self.uh


class Linearsolver_activation:
    def __init__(self, a, loc_loss_form, loc_loss, D_Ih, act, psi, bcs):
        self.adj_compiled = fem.form(a)
        self.L_compiled = fem.form(loc_loss_form)
        self.dL_compiled = fem.form(D_Ih)
        self.Adj = fem.create_matrix(self.adj_compiled)
        self._L = loc_loss
        self.dL = fem.create_matrix(self.dL_compiled)
        self.bcs = bcs
        self._Adj_scipy = self.Adj.to_scipy()
        self._dL_scipy = self.dL.to_scipy()
        self._psi = psi
        self._act = act
    
    def solve(self):
        self._Adj_scipy.data[:] = 0
        self._L.x.array[:] = 0
        self._dL_scipy.data[:] = 0
        fem.assemble_matrix(self.Adj, self.adj_compiled, bcs=self.bcs)
        fem.assemble_matrix(self.dL, self.dL_compiled, bcs=self.bcs)
        fem.assemble_vector(self._L.x.array, fem.form(self.L_compiled))
        
        
        vec = self._act.dx(self._L.x.array) * self._dL_scipy

        fem.apply_lifting(vec, [self.adj_compiled], [self.bcs])
        [bc.set(vec) for bc in self.bcs]

        Adj_inv = scipy.sparse.linalg.splu(self._Adj_scipy)
        self._psi.x.array[:] = Adj_inv.solve(vec)
        return self._psi

    
class loss_act_class:
    def __init__(self, t0):
        self.t0 = t0
    def _z(self, x):
        return x/self.t0
    def __call__(self,x):
        z = self._z(x)
        return np.where(x>self.t0, 1, 1/2*z**4-z**3-1/2*z**2+2*z)
    def dx(self,x):
        z = self._z(x)
        return np.where(x>self.t0, 0, 2*z**3-3*z**2-z+2)/self.t0


class SUPG_solver:
    def __init__(self, pde_data):

        #data for FEM-solution-space
        self.domain,self.Wh,self.uh,self.eps,self.b,self.c,self.f,self.g,self.bcs = pde_data
        self._u= ufl.TrialFunction(self.Wh)
        self._v = ufl.TestFunction(self.Wh)

        #FEM space for the SUPG-parameters/weights
        self.Yh = functionspace(self.domain, ("DG", 0))
        self.yh = fem.Function(self.Yh)

        #number of locally owned cells
        self._owned_cells = self.yh.x.index_map.size_local

        #SUPG-forms
        self._a = self.eps * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx + ufl.dot(self.b, ufl.grad(self._u)) * self._v * ufl.dx
        sh_test = (self.yh * ufl.dot(self.b, ufl.grad(self._v))) 
        self._sh = -self.eps * ufl.div(ufl.grad(self._u)) * sh_test * ufl.dx + ufl.dot(self.b, ufl.grad(self._u)) * sh_test * ufl.dx
        if self.c != None:
            self._sh += self.c * self._u* sh_test * ufl.dx
            self._a +=  self.c * self._u* self._v * ufl.dx

        self._L = self.f * self._v * ufl.dx
        if self.g is not None:
            self._L += self.g * self._v * ufl.ds
        self._rh = self.f * self.yh * ufl.dot(self.b, ufl.grad(self._v)) * ufl.dx

        # 1st LinearProblem
        self.prblm = LinearSolver(
            a=self._a + self._sh, 
            L=self._L+self._rh, 
            bcs=self.bcs, 
            uh=self.uh
        )
        self.prblm.solve()

    def set_weights(self, weights):
        #set weights for locally owned dofs
        self.yh.x.array[:self._owned_cells] = weights[:self._owned_cells]
        self.yh.x.scatter_forward()

        self.prblm.solve()


class SUPG_grad_adjoint_method_solver(SUPG_solver):
    def __init__(self, pde_data, loss_form):
        super().__init__(pde_data)
        #The adjoint problem
        
        #the adjoint problem needs homgenous boundary conditions.
        dbc_dofs = np.array([], dtype=np.int32)
        for bc in self.bcs:
            dbc_dofs = np.append(dbc_dofs, bc.dof_indices()[0])
        self._hom_bcs = [fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), dbc_dofs, self.Wh)]

        self._Rh = ufl.replace(self._a+self._sh - self._L-self._rh, {self._u:self.uh})
        self._loc_loss_form = loss_form
        self._loss_form = loss_form
        for integral in self._loc_loss_form.integrals():
            integrand = integral.integrand()
            self._loc_loss_form = ufl.replace(self._loc_loss_form, {integrand: integrand * ufl.TestFunction(self.Yh)})
        self._local_loss = fem.Function(self.Yh)
        fem.assemble_vector(self._local_loss.x.array, fem.form(self._loc_loss_form))

        #forms for the adjoint problem
        Rh_w = ufl.derivative(form=self._Rh, coefficient=self.uh, argument=self._u)
        self._D_Ih = ufl.derivative(form=loss_form, coefficient=self.uh)
        self._psi = fem.Function(self.Wh)
        self._adjoint_bilin = ufl.replace(ufl.adjoint(Rh_w), {self.uh:self._v})
        # 2nd LinearProblem
        self.adj_prblm = LinearSolver(
            a=self._adjoint_bilin, 
            L=self._D_Ih, 
            bcs=self._hom_bcs, 
            uh=self._psi
        )
        self.adj_prblm.solve()


        #The gradient problem

        #forms for the gradient computation
        z = ufl.TrialFunction(self.Yh)
        self._Rh_y = ufl.action(ufl.adjoint(ufl.derivative(form=-self._Rh, coefficient=self.yh, argument=z)), self._psi)
        self._grd = fem.Function(self.Yh)
        fem.assemble_vector(self._grd.x.array, fem.form(self._Rh_y))

        norm_b = ufl.sqrt(ufl.dot(self.b,self.b))
        h = ufl.CellDiameter(domain=self.domain) 
        alpha = norm_b*h/(2*self.eps)
        Xi = (1/ufl.tanh(alpha)-1/alpha)
        tau_K = h/(2*norm_b)*Xi
        Th = fem.functionspace(self.domain, ('DG', 0))
        tau = fem.Function(Th)
        tau_exp = fem.Expression(tau_K, Th.element.interpolation_points())
        tau.interpolate(tau_exp)
        self.set_weights(tau.x.array)


        self._upper = 100*self.yh.x.array

        self._bounds = scipy.optimize.Bounds(lb=np.zeros_like(self._upper), ub=self._upper)

    def set_weights(self, weights):
        super().set_weights(weights=weights)

        self.adj_prblm.solve()

        self._local_loss.x.array[:] = 0
        self._grd.x.array[:] = 0
        fem.assemble_vector(self._local_loss.x.array, fem.form(self._loc_loss_form))
        fem.assemble_vector(self._grd.x.array, fem.form(self._Rh_y))

    def local_loss(self):
        return self._local_loss.x.array

    def loss(self):
        return self._local_loss.x.array.sum()
        

    def grad(self):
        return self._grd.x.array

    def _callback(self, intermediate_result):
        fval = intermediate_result.fun
        print(f"J: {fval}")


    def _eval(self, weights):
        self.set_weights(weights=weights)
        return self.loss()
    

    def _eval_grad(self, weights):
        return self.grad()
    

    def optimize(self, algorithm='L-BFGS-B', ftol=1e-16, gtol=1e-16, max_iter=10000):
        scipy.optimize.minimize(
            fun=self._eval,
            x0=self.yh.x.array,
            jac=self._eval_grad,
            method=algorithm,
            callback=lambda intermediate_result: print(f"J: {intermediate_result.fun}"),
            bounds=self._bounds,
            options={'ftol':ftol, 'gtol':gtol, 'maxiter':max_iter}
        )
    

class SUPG_grad_activation_solver(SUPG_grad_adjoint_method_solver):
    def __init__(self, pde_data, loss_form, t0):
        self._loss_act = loss_act_class(t0)
        super().__init__(pde_data, loss_form=loss_form)
        # 2nd LinearProblem
        self._D_Ih = ufl.derivative(form=self._loc_loss_form, coefficient=self.uh)
        self.adj_prblm = Linearsolver_activation(
            a=self._adjoint_bilin, 
            loc_loss_form=self._loc_loss_form,
            loc_loss=self._local_loss,
            D_Ih=self._D_Ih, 
            act = self._loss_act,
            bcs=self._hom_bcs, 
            psi=self._psi
        )
        self.adj_prblm.solve()

    def local_loss(self):
            return self._loss_act(self.adj_prblm._L.x.array)
