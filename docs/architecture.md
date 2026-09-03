# Architecture

The package is organized by responsibility rather than by experiment.

| Package | Responsibility |
| --- | --- |
| `fem` | DOLFINx interpolation, assembly, field sampling, and linear solves |
| `supg` | SUPG state solve, objectives, adjoint gradient, optimization |
| `benchmarks` | Reproducible named PDE definitions and registry |
| `stabilization` | Standard SUPG, Tabata upwinding, and AFC variants |
| `graph` | Mesh adjacency and node/edge feature construction |
| `data` | Dataset schema, persistence, and loaders |
| `models` | Configurable MLP/GNN/attention models and output constraints |
| `autograd` | Differentiable adapters around FEM objectives |
| `training` | Supervised and physics-informed training loops |
| `optim` | SciPy adapters for PyTorch parameters |
| `viz` | Optional field and comparison plotting |

Dependencies point from high-level workflows toward numerical primitives.
In particular, FEM and benchmark modules never import training datasets, and
importing `supgml` never reads files or constructs a dataset.
