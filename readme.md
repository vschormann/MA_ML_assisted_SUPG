# Readme
This repository stores code for my masters thesis to create machine learning models for optimizing SUPG-parameters for singularly perturbed convection diffusion problems. [Acknowledgements](#acknowledgements) for the frameworks used are at the end of the file. Here is a short description.

FEM-routines are implemented using FEniCsx - in particular [Dolfinx](doi.org/10.5281/zenodo.10447666) and [UFL](https://dl.acm.org/doi/10.1145/2566630). Additional computations are done using [NumPy](https://www.nature.com/articles/s41586-020-2649-2). Visualizations use [Pyvista](https://joss.theoj.org/papers/10.21105/joss.01450) and [Matplotlib](https://ieeexplore.ieee.org/document/4160265).

[Pytorch](https://arxiv.org/abs/1912.01703v1) is used to implement the Neutral Networks.


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

# Tests
Files that provide a sanity check for the modules listed above. 
1. [Test_1_gradcheck](/Test_1_gradcheck.py) compares the gradient computed with dolfinx to a numerically computed gradient using torch.autograd.gradcheck ten times. Note that this requires the device to be set to cpu and the dtype to double to maximize precision. Still out of the ten tests implemented usually 3 or more will fail even though absolute tolerance is set to 1e-2.

2. [Test_2_compare_Fenicsx_Pytorch](/Test_2_compare_FenicsX_Pytorch.py) implements a simple gradients descent using a supg.data object and NumPy and compares its loss values and solution to a Pytorch model that uses a the supg_loss function from [supg_torch](/torch_classes/supg_torch.py), a trainable vector and torch.optim.SGD to make sure the Pytorch classes work as expected.

# Examples
Codes that experiment with the models, create visualizations in form of images, gifs and mp4-videos.

# Acknowledgements

## Dolfinx
> I. A. Baratta, J. P. Dean, J. S. Dokken, M. Habera, J. S. Hale, C. N. Richardson, M. E. Rognes, M. W. Scroggs, N. Sime, and G. N. Wells. DOLFINx: The next generation FEniCS problem solving environment, preprint (2023). [[doi.org/10.5281/zenodo.10447666]](doi.org/10.5281/zenodo.10447666)

## Basix
Basix is the finite element backend of FEniCSx, responsible for generating finite element basis functions.
> M. W. Scroggs, J. S. Dokken, C. N. Richardson, and G. N. Wells. Construction of arbitrary order finite element degree-of-freedom maps on polygonal and polyhedral cell meshes, ACM Transactions on Mathematical Software 48(2) (2022) 18:1–18:23. [[arΧiv]](https://arxiv.org/abs/2102.11901) [[doi.org/10.1145/3524456]](https://dl.acm.org/doi/10.1145/3524456)

> M. W. Scroggs, I. A. Baratta, C. N. Richardson, and G. N. Wells. Basix: a runtime finite element basis evaluation library, Journal of Open Source Software 7(73) (2022) 3982. [[doi.org/10.21105/joss.03982]](https://joss.theoj.org/papers/10.21105/joss.03982)

## UFL
> M. S. Alnaes, A. Logg, K. B. Ølgaard, M. E. Rognes and G. N. Wells. Unified Form Language: A domain-specific language for weak formulations of partial differential equations, ACM Transactions on Mathematical Software 40 (2014). [[arΧiv]](https://arxiv.org/abs/1211.4047) [[doi.org/10.1145/2566630]](https://dl.acm.org/doi/10.1145/2566630)

## Pytorch
> Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from [http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)

## Pyvista
> Sullivan et al., (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). Journal of Open Source Software, 4(37), 1450, [https://doi.org/10.21105/joss.01450](https://joss.theoj.org/papers/10.21105/joss.01450)

## Numpy
> Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: [10.1038/s41586-020-2649-2](https://www.nature.com/articles/s41586-020-2649-2). [(Publisher link)](https://www.nature.com/articles/s41586-020-2649-2).

## Matplotlib
> [J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.](https://ieeexplore.ieee.org/document/4160265)