# Tutorial 1: from a convection–diffusion equation to SUPG

This tutorial is the numerical starting point for the repository. It explains
what is being predicted before introducing a neural network.

## Problem

We consider a stationary convection–diffusion equation on a domain
\(\Omega\):

\[
-\varepsilon\Delta u + \boldsymbol b\cdot\nabla u = f.
\]

When diffusion `eps` is small compared with the velocity `b`, the solution can
have thin layers. A standard continuous Galerkin discretisation may then
oscillate. SUPG adds a cellwise streamline term weighted by a parameter
\(\tau_K\). In this project, those parameters are first optimized and later
predicted from local FEM/graph information.

## Build the finite-element ingredients

The following is intentionally ordinary DOLFINx/UFL code. It makes the mesh,
space, coefficients, and boundary condition inspectable.

```python
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from mpi4py import MPI

mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 24, 24)
V = fem.functionspace(mesh, ("CG", 1))
u_h = fem.Function(V, name="state")
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

eps = fem.Constant(mesh, default_scalar_type(1e-3))
b = ufl.as_vector((fem.Constant(mesh, 1.0), fem.Constant(mesh, 0.2)))
f = fem.Constant(mesh, default_scalar_type(1.0))
```

The Galerkin contribution is

```python
a_galerkin = (
    eps * ufl.dot(ufl.grad(u), ufl.grad(v))
    + ufl.dot(b, ufl.grad(u)) * v
) * ufl.dx
L_galerkin = f * v * ufl.dx
```

## Add a cellwise stabilization field

The parameter field is DG0: there is one value per mesh cell. This is exactly
the quantity represented by graph nodes later in the workflow.

```python
Y = fem.functionspace(mesh, ("DG", 0))
tau = fem.Function(Y, name="tau")
streamline_test = tau * ufl.dot(b, ufl.grad(v))
```

The full assembly and solve are packaged so that every benchmark uses the same
well-tested mechanics. The scientific choices—the equation, coefficients,
boundary data, objective, and interpretation—stay visible in the tutorial and
canonical notebook.

## Optimize with a discrete adjoint

Create a `ConvectionDiffusionProblem` and choose an objective, for example an
integrated squared state. `AdjointSUPGSolver` solves the state problem, forms
the discrete adjoint, and returns one gradient entry per DG0 cell.

```python
from supgml.supg import AdjointSUPGSolver, ConvectionDiffusionProblem

problem = ConvectionDiffusionProblem(mesh, V, u_h, eps, b, None, f, None, [bc])
objective = u_h**2 * ufl.dx
solver = AdjointSUPGSolver(problem, objective)
solver.set_weights(tau.x.array)

print(solver.loss())
gradient = solver.grad()
```

Continue in [Tutorial 2](graphs.md) to turn this FEM state and optimized
parameter field into a graph-learning case. The executable counterpart is
`notebooks/01_supg_objectives.ipynb`.
