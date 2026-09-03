# Tutorial 3: learn SUPG parameters

Both learning approaches predict one DG0 value per graph node; they differ in
how the prediction is judged.

## Supervised learning

The target is the direct-adjoint optimized parameter field in `graph.y`. This
permits a controlled comparison of MLP, GCN, GraphSAGE, GAT, and GATv2.

```python
from supgml.models import BoundedOutput, create_model

base = create_model(
    "gatv2", in_channels=graph.x.shape[1], hidden_channels=32,
    num_layers=4, edge_dim=graph.edge_attr.shape[1],
)
model = BoundedOutput(base, upper="upper")
prediction = model(graph)
```

The output constraint is numerical modelling, not decoration: it keeps
predictions in the admissible SUPG interval. A low parameter MSE is useful but
does not by itself establish an oscillation-free state solution.

## FEM-backed learning

Self-supervised training passes the prediction to the FEM objective. The custom
autograd bridge uses a DOLFINx discrete-adjoint gradient in PyTorch's backward
pass instead of differentiating through an opaque sparse solve.

```python
from supgml.autograd import FEMObjective

fem_objective = FEMObjective(solver)
loss = fem_objective(model(graph))
loss.backward()
```

Validate every new FEM objective against finite differences. Run the documented
experiment matrices with `supgml-train experiments/ch4_supervised.json` and
`supgml-train experiments/ch4_self_supervised.json`.

Continue with [Tutorial 4](revised-study.md) for the Chapter 5 revision.
