import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import numpy as np

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
        self.prblm = LinearProblem(a=a + sh, L=L+rh, bcs=bcs, u=self.uh)
        self.prblm.solve()


        #The adjoint problem
        
        #the adjoint problem needs homgenous boundary conditions.
        dbc_dofs = np.array([], dtype=np.int32)
        for bc in bcs:
            dbc_dofs = np.append(dbc_dofs, bc.dof_indices()[0])
        hom_bcs = [fem.dirichletbc(fem.Constant(domain, default_scalar_type(0.0)), dbc_dofs, Wh)]

        #the trial function needs to be substituted with a coefficient function in order for symbolic differentiation to work.
        self.w = ufl.Coefficient(Wh) 
        self.Rh = ufl.replace(a+sh - L-rh, {u:self.w})
        self.loss_form = loss_form

        #forms for the adjoint problem
        Rh_w = ufl.derivative(form=self.Rh, coefficient=self.w, argument=u)
        D_Ih = ufl.derivative(form=self.loss_form, coefficient=self.uh, argument=v)
        self.psi = fem.Function(Wh)
        # 2nd LinearProblem
        self.adj_prblm = LinearProblem(a=ufl.adjoint(Rh_w), L=D_Ih, bcs=hom_bcs, u=self.psi)
        self.adj_prblm.solve()



        #The gradient problem
        
        #volume of the cells needs to be factored out for gradient computation
        y = ufl.TestFunction(Yh)
        cvol = fem.assemble_vector(fem.form(y * ufl.dx)).array
        vol_fn = fem.Function(Yh)
        vol_fn.x.array[:len(cvol)] = 1/cvol
        vol_fn.x.scatter_forward()

        #forms for the gradient computation
        z = ufl.TrialFunction(Yh)
        self.grd_fn = fem.Function(Yh)
        Rh_y = ufl.replace(ufl.derivative(form=self.Rh, coefficient=self.yh, argument=y), {self.w:self.uh, v:self.psi})

        # 3rd LinearProblem
        self.grd_prblm = LinearProblem(a=y * vol_fn * z * ufl.dx, L=-Rh_y, bcs=[], u=self.grd_fn)
        self.grd_prblm.solve()


    def set_weights(self, weights):
        #set weights for locally owned dofs
        self.yh.x.array[:self.owned_cells] = weights[:self.owned_cells]
        self.yh.x.scatter_forward()

        #reassemble matrices and vectors and compute solutions for the LinearProblems
        self.prblm.A.assemble()
        self.prblm.b.assemble()
        self.prblm.solve()

        self.adj_prblm.A.assemble()
        self.adj_prblm.b.assemble()
        self.adj_prblm.solve()

        self.grd_prblm.b.assemble()
        self.grd_prblm.solve()
    
    def loss(self):
        return fem.assemble_scalar(fem.form(self.loss_form))

    def grad(self):
        return self.grd_fn.x.array[:self.owned_cells]
    


