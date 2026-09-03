# Tutorial 2: turn FEM cases into graph-learning data

Each graph has a numerical meaning: a node is a finite-element cell, an edge
connects neighboring cells, and the target is a cellwise SUPG parameter. This
page explains that contract before model training.

## Decide what one case represents

A case combines a named PDE benchmark, a mesh, its standard-SUPG state, a
direct-adjoint optimized DG0 parameter field, and IDs that recover the FEM
solver later. The Chapter 4 dataset varies benchmark and mesh deliberately;
record that matrix in the canonical notebook, not in copied training scripts.

## Build the graph

`GraphBuilder` converts a configured solver into a PyTorch Geometric data
object. It stores feature names and a schema version so that a model never has
to guess the meaning or order of its input columns.

```python
from supgml.graph import GraphBuilder

graph = GraphBuilder(include_edge_features=True).build(
    solver,
    target=optimized_tau,
    upper=upper_bounds,
    problem_id=problem_id,
    mesh_id=mesh_id,
)

print(graph.feature_names)
print(graph.x.shape, graph.edge_index.shape, graph.y.shape)
```

The standard features include coefficients, cell diameter, and the standard
SUPG solution and derivatives. Name and version any new feature before
regenerating cases.

## Persist and validate

Use `CaseRepository` for the on-disk convention, then check split, case count,
feature order, target/upper shapes, and mesh IDs before training.

```python
from supgml.data import CaseRepository

repository = CaseRepository("data")
# repository.save(case_id, solver, graph, split="train")
```

The executable counterpart is `notebooks/02_ch4_generate_dataset.ipynb`.
Continue with [Tutorial 3](learning.md) for supervised and FEM-backed learning.
