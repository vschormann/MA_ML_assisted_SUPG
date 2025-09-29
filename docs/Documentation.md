# Modules
Currently there are two modules in this project. One to handle FEM-computations using FEniCsx and one to bridge FEniCsx and Pytorch.

## [supg](/supg/) 
This module stores a [class](/supg/supg.py) that stores solvers for the SUPG-approximation, the gradient with respect to the SUPG-parameters, the loss value and a local loss on the individual cells. The object can be initialized with the following data
- domain: dolfinx.mesh.Mesh
- Wh: dolfinx.fem.FunctionSpace
- Coefficients for eps, b, c, f (any input that ufl treats as a coefficient ufl.Constant, dolfinx.fem.Function, ...)
- bcs: a list of dolfinx.fem.DirichletBC
- boundary_eval: optional boolean to toggle evaluation of the loss function at the Dirichlet boundary. 

It stores instances of the dolfinx.fem.petsc.LinearProblem and their solutions which are updated by the function set_weights. If None it is set to True.

Some example problems are stored in tuples in an [extra file](/supg/sp_problems.py) and can be imported as dat+\[nr]. For example: an supg.data object can be initialized with:

```Python
from supg import supg
from supg.sp_problems import dat1 as dat

sd = supg.data(*dat, False)
```

The module also contains a [convenience class](/supg/plotter.py) that stores a pyvista.UnstructuredGrid created from a dolfinx.fem.FunctionSpace to simplify visualization. 

There is additional functionality for tabulating and visualizing functions on facets to evaluate jumps. But currently it isn't used in any of the routines and might be removed or moved to a different file later.

## [torch_classes](/torch_classes/)
This module creates a bridge between Pytorch and Dolfinx by subclassing torch.autograd.Function. The [object](/torch_classes/supg_torch.py) takes 
- a supg.data-object and
- a torch.Tensor

where the forward pass takes the global_loss value from the supg.data object and the backward pass the constrained_gradient. The supg_loss function takes the same arguments and can be used in Pytorch routines.

> Note that all PDE-related computations including the symbolic differentiations needed to compute the gradient are done in Dolfinx and UFL. Pytorch.autograd mechanics are only used when propagating gradients backwards in a neural network.

The module also stores different instances of subclasses of torch.nn.Module as neural network models.