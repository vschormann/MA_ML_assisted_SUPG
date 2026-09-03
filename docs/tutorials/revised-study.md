# Tutorial 4: the revised AFC-BJK study

Chapter 5 is a methodological change, not a larger Chapter 4 sweep. The
reference becomes an AFC approximation with BJK limiter, and model selection is
based on the FEM loss relative to that reference.

## Define the revised reference

For SPDE 3, define the mesh, convection direction, and discontinuous boundary
data in `notebooks/06_ch5_build_afc_target.ipynb`. Compute or load the AFC-BJK
solution \(u_{\mathrm{BJK}}\), then set

\[
J(\tau) = \int_\Omega (u_\tau-u_{\mathrm{BJK}})^2\,dx.
\]

```python
from supgml.supg import AdjointSUPGSolver

objective = (u_h - u_bjk)**2 * ufl.dx
adjoint_solver = AdjointSUPGSolver(problem, objective)
```

The flux-correction limiter is reusable, so it lives in
`supgml.stabilization`; the SPDE and rationale remain in the notebook.

## Train and select models

The revised study compares a wide MLP and GATv2. It uses interior-node masking,
a combined Huber/L2 target loss, cosine warm restarts, and periodic FEM-loss
evaluation. The selected checkpoint has the lowest AFC-BJK FEM loss, not merely
the lowest training loss.

```bash
supgml-train experiments/ch5_revised.json
```

## Interpret carefully

Parameter perturbations and a local-predictor lower bound show why optimized
parameter targets can be non-unique and why long-range graph information may
help. Compare target loss, FEM loss, fields, and line plots together before
drawing conclusions.

Use notebooks 06–08 for the executable study and notebook 09 for reproducible
reported figures.
