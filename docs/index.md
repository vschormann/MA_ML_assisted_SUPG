# SUPG-ML

`supgml` is the readable, installable form of the reusable numerical and
machine-learning code developed for the thesis on learning SUPG parameters for
singularly perturbed convection–diffusion problems.

## Start here

The repository is best read as a short scientific workflow, not as a collection
of unrelated notebooks:

1. **Define and stabilize a PDE.** Notebook 01 shows the mesh, finite-element
   spaces, weak form, cellwise SUPG parameter, objective, and discrete adjoint.
2. **Create graph learning cases.** Notebook 02 maps FEM cells and fields to a
   documented graph schema, with optimized cellwise parameters as targets.
3. **Compare Chapter 4 models.** Notebooks 03–05 compare MLP, GCN, GraphSAGE,
   GAT, and GATv2 using supervised and adjoint-backed objectives.
4. **Follow the Chapter 5 revision.** Notebooks 06–08 replace the reference
   solution with AFC-BJK, train the revised MLP/GATv2 models, and assess target
   ambiguity and non-local information.
5. **Regenerate reported figures.** Notebook 09 is a read-only reporting step
   with explicit figure provenance.

See the [canonical notebook sequence](notebooks.md) for the full map and the
repository's `notebooks/README.md` for execution order.

## What belongs where

The notebooks retain the scientific choices: SPDE definitions, boundary data,
weak forms, objectives, experiment matrices, and interpretation. The `supgml`
package contains the mechanics that would otherwise be copied between
experiments: DOLFINx assembly and solves, the discrete-adjoint implementation,
the PyTorch/FEniCSx autograd bridge, graph serialization, model factories,
training loops, and visualisation helpers.

This boundary makes a result inspectable without making every notebook a copy
of numerical infrastructure.

## Choose a workflow

| Goal | Begin with | Main package areas |
| --- | --- | --- |
| Understand SUPG and the adjoint | Notebook 01 | `supgml.supg`, `supgml.fem`, `supgml.stabilization` |
| Recreate Chapter 4 data/models | Notebooks 02–05 | `supgml.graph`, `supgml.data`, `supgml.models`, `supgml.training` |
| Recreate the revised study | Notebooks 06–08 | `supgml.stabilization`, `supgml.autograd`, `supgml.experiments` |
| Reuse a component in new work | [API guide](api.md) | the relevant `supgml.*` subpackage |

## Installation and provenance

Install the repository into the active DOLFINx/Jupyter environment before
opening the canonical notebooks:

```bash
python -m pip install -e '.[ml,viz]'
```

The [installation guide](installation.md) covers the DOLFINx environment and
the macOS PyTorch/OpenMP kernel issue. Commit
`aed55ecdaf7c99b4f0f89662e48eab106de8013f` records the repository at thesis
submission; later commits are readability and reuse refactorings. Submitted and
abandoned exploratory notebooks are retained in the archive rather than being
presented as final workflows.
