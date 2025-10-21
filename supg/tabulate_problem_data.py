from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from dolfinx import default_scalar_type

def tabulate_problem_data(domain, pde_data):
    Yh = fem.functionspace(domain, ("DG", 0))

    Wh,eps,b,c,f,bcs,g = pde_data(domain)
    y = ufl.TestFunction(Yh)
    z = ufl.TrialFunction(Yh)
    func = fem.Function(Yh)
    
    dat = [eps, b[0], b[1], c, f, g]

    lhs = y * ufl.dx
    retlist = [Yh.tabulate_dof_coordinates()[:,:domain.topology.dim]]
    datlist = []

    for coeff in dat:
        if coeff is None:
            coeff = fem.Constant(domain, default_scalar_type(0.0))
        lhs = coeff * y * ufl.dx
        problem = LinearProblem(a=z * y * ufl.dx, L=lhs)
        func = problem.solve()
        datlist.append(func.x.array.reshape(-1,1))
    
    retlist.append(np.array(datlist))

    bcs_ind = bcs[0].dof_indices()[0]
    retlist.append(Wh.tabulate_dof_coordinates()[bcs_ind,:domain.topology.dim])

    try:
        retlist.append(bcs[0].g.x.array[bcs_ind].reshape(-1,1))
    except:
        val = bcs[0].g.value
        arr = val*np.ones(bcs_ind.size)
        retlist.append(arr.reshape(-1,1))
    
    return retlist