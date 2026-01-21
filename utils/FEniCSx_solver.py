import ufl
from dolfinx import default_scalar_type, fem, la
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
    

class FEniCSx_solver:
    def __init__(self, pde_data, loss_form):

        #data for FEM-solution-space
        domain,Wh,self.uh,eps,b,c,f,g,bcs = pde_data
        u = ufl.TrialFunction(Wh)
        v = ufl.TestFunction(Wh)

        #FEM space for the SUPG-parameters/weights
        Yh = functionspace(domain, ("DG", 0))
        self.yh = fem.Function(Yh)

        #number of locally owned cells
        self.owned_cells = self.yh.x.index_map.size_local

        #SUPG-forms
        a = eps * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(b, ufl.grad(u)) * v * ufl.dx
        sh_test = (self.yh * ufl.dot(b, ufl.grad(v))) 
        sh = -eps * ufl.div(ufl.grad(u)) * sh_test * ufl.dx + ufl.dot(b, ufl.grad(u)) * sh_test * ufl.dx
        if c != None:
            sh += c * u * sh_test * ufl.dx
            a +=  c * u * v * ufl.dx

        L = f * v * ufl.dx
        if g is not None:
            L += g * v * ufl.ds
        rh = f * self.yh * ufl.dot(b, ufl.grad(v)) * ufl.dx

        # 1st LinearProblem
        self.prblm = LinearSolver(
            a=a + sh, 
            L=L+rh, 
            bcs=bcs, 
            uh=self.uh
        )
        self.prblm.solve()


        #The adjoint problem
        
        #the adjoint problem needs homgenous boundary conditions.
        dbc_dofs = np.array([], dtype=np.int32)
        for bc in bcs:
            dbc_dofs = np.append(dbc_dofs, bc.dof_indices()[0])
        hom_bcs = [fem.dirichletbc(fem.Constant(domain, default_scalar_type(0.0)), dbc_dofs, Wh)]

        #the trial function needs to be substituted with a coefficient function in order for symbolic differentiation to work.
        self.w = ufl.Coefficient(Wh) 
        self.Rh = ufl.replace(a+sh - L-rh, {u:self.uh})
        self.loss_form = loss_form

        #forms for the adjoint problem
        Rh_w = ufl.derivative(form=self.Rh, coefficient=self.uh, argument=u)
        D_Ih = ufl.derivative(form=self.loss_form, coefficient=self.uh, argument=v)
        self.psi = fem.Function(Wh)
        adjoint_bilin = ufl.replace(ufl.adjoint(Rh_w), {self.uh:v})
        # 2nd LinearProblem
        self.adj_prblm = LinearSolver(
            a=adjoint_bilin, 
            L=D_Ih, 
            bcs=hom_bcs, 
            uh=self.psi
        )
        self.adj_prblm.solve()


        #The gradient problem

        #forms for the gradient computation
        z = ufl.TrialFunction(Yh)
        self.grd_fn = fem.Function(Yh)
        self.Rh_y = ufl.action(ufl.adjoint(ufl.derivative(form=-self.Rh, coefficient=self.yh, argument=z)), self.psi)


    def set_weights(self, weights):
        #set weights for locally owned dofs
        self.yh.x.array[:self.owned_cells] = weights[:self.owned_cells]
        self.yh.x.scatter_forward()

        self.prblm.solve()

        self.adj_prblm.solve()
    
    def loss(self):
        return fem.assemble_scalar(fem.form(self.loss_form))

    def grad(self):
        return fem.assemble_vector(fem.form(self.Rh_y)).array
    


