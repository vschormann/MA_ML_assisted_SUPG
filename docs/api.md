# Public API

## Benchmarks and solvers

```python
from supgml.benchmarks import create

solver = create("wedge", mesh=mesh, eps_val=1e-8)
solver.set_weights(parameters)
loss = solver.loss()
gradient = solver.grad()
result = solver.optimize()
```

Benchmarks accept readable names. Integer IDs remain supported only for old
datasets and notebooks.

## Graph construction

```python
from supgml.graph import GraphBuilder

graph = GraphBuilder(include_edge_features=True).build(
    solver,
    target=optimized_parameters,
    upper=upper_bounds,
    problem_id=0,
    mesh_id=0,
)
```

Graphs declare a `schema_version` and `feature_names`; model code should derive
`in_channels` from `graph.x.shape[1]` rather than assume a fixed feature count.

## Models and training

```python
from supgml.models import BoundedOutput, create_model
from supgml.training import fit

base = create_model(
    "gatv2",
    in_channels=graph.x.shape[1],
    hidden_channels=32,
    num_layers=4,
    edge_dim=graph.edge_attr.shape[1],
)
model = BoundedOutput(base, upper="upper")
history = fit(model, loader, optimizer, epochs=100, device=device)
```

## Differentiable FEM loss

```python
from supgml.autograd import BatchedFEMLoss

loss_fn = BatchedFEMLoss({graph.mesh_id: solver})
loss = loss_fn(batch.ptr, batch.mesh_id, model(batch))
```

The backward pass uses the adjoint gradient returned by each solver. Validate
new solver objectives against finite differences before training.
