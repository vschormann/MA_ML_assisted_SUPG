import random
from supg import supg
from mpi4py import MPI
from dolfinx import mesh as msh
from dolfinx import fem
from dolfinx import default_scalar_type
import ufl
import torch_classes.nn_models as nn_models
import torch
from torch_classes.supg_torch import supg_loss
import numpy as np

def tensor1(eps=1e-8,b1=1,b2=0,c=0,f=1,bcs=0,h=100,w=100):
    eps = eps*torch.ones((h,w))
    b1 = b1*torch.ones((h,w))
    b2 = b2*torch.zeros((h,w))
    c = c*torch.zeros((h,w))
    f = f*torch.ones((h,w))
    retval = torch.stack([eps,b1,b2,c,f])
    retval[:3,:,0] = 0
    retval[:3,:,-1] = 0
    retval[:3,0,:] = 0
    retval[:3,-1,:] = 0
    retval[3,:,0] = 1
    retval[3,:,-1] = 1
    retval[3,0,:] = 1
    retval[3,-1,:] = 1
    retval[4,:,0] = 0
    retval[4,:,-1] = 0 
    retval[4,0,:] = 0
    retval[4,-1,:] = 0
    return retval

lr=1e-4

for i in range(10):
    eps_val=1e-8+random.normalvariate(mu=0, sigma=1e-9)
    b1_val=1+random.normalvariate(mu=0, sigma=1e-3)
    b2_val=random.normalvariate(mu=0, sigma=1e-3)
    c_val=0
    f_val=1

    def pde_data(domain):
        Wh = fem.functionspace(domain, ('P', 2))
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        boundary_facets = msh.exterior_facet_indices(domain.topology)
        boundary_dofs = fem.locate_dofs_topological(Wh, domain.topology.dim-1, boundary_facets)
        eps = fem.Constant(domain, default_scalar_type(eps_val))
        b = ufl.as_vector((fem.Constant(domain, default_scalar_type(b1_val)),fem.Constant(domain, default_scalar_type(b2_val))))
        c = fem.Constant(domain, default_scalar_type(c_val))
        f = fem.Constant(domain, default_scalar_type(f_val))
        bcs = [fem.dirichletbc(fem.Constant(domain, default_scalar_type(0.0)), boundary_dofs, Wh)]
        return Wh,eps,b,c,f,bcs,None


    tensor=tensor1(
        eps=eps_val,
        b1=b1_val,
        b2=b2_val,
        c=0,
        f=1,
        bcs=0,
        h=20,
        w=20)
    torch.save(tensor, 'data/inputs/20x20/ex1_'+str(i)+'.pt')

    domain = msh.create_unit_square(MPI.COMM_WORLD, 20, 20, msh.CellType.quadrilateral)

    sd=supg.data(domain, pde_data, boundary_eval=False)

    pts = sd.Yh.tabulate_dof_coordinates()
    x = pts[:,0]
    y = pts[:,1]
    shape=np.lexsort((pts[:,0], -pts[:,1])).reshape(-1,20)


    model = nn_models.md1(torch.Tensor(sd.yh.x.array).view(1,-1))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=10, history_size=10)
    loss_array=np.array([])
    def closure():
        optimizer.zero_grad()
        pred = model()
        loss = supg_loss(sd, pred)
        loss.backward()
        return loss

    for t in range(100):
        optimizer.step(closure)

    weights=torch.Tensor(sd.yh.x.array)[shape].view(1,20,20)

    torch.save(weights, 'data/targets/20x20/ex1_'+str(i)+'.pt')