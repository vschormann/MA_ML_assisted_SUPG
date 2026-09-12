# Machine-learning-assisted SUPG

This repository accompanies the master's thesis *Machine learning assisted
optimization of stabilization parameters for convection-diffusion-reaction
equations*. It investigates whether local finite-element data and graph message
passing can predict one SUPG stabilization parameter per mesh cell.

The submitted experiments support a qualified conclusion: adjacent-cell
information can help on the heterogeneous benchmark set, and GATv2 had the best
overall generalization, but no model consistently produced physical solutions
for every problem. Chapter 5's focused AFC-BJK study improved the supervised
results while retaining outflow artefacts.

> **Research provenance.** Commit
> [`aed55ecdaf7c99b4f0f89662e48eab106de8013f`](https://github.com/vschormann/MA_ML_assisted_SUPG/commit/aed55ecdaf7c99b4f0f89662e48eab106de8013f)
> records the repository at thesis submission. Later commits reorganize,
> document, and test the work; they were made with assistance from OpenAI
> Codex and are not part of the version submitted for assessment.

## Start here

- Read [Results and research path](docs/results.md) for the shortest account of
  the question, experiments, negative results, and conclusions.
- Use the [guided notebook sequence](notebooks/README.md) to follow the thesis
  argument, or open the [rendered notebooks](docs/rendered-notebooks/index.html)
  without a Python environment.
- Read [Continuing this work](docs/continuation.md) before attempting a new run.
  It records which artefacts are tracked, what a fresh clone can reproduce, and
  how to keep new work separate from the submitted evidence.
- Browse the root [`index.html`](index.html) for the public overview and the
  committed 88-training/14-test-case DOLFINx gallery.

## Repository map and execution status

| Path | Purpose and status |
| --- | --- |
| `notebooks/archive/` | Submitted and exploratory notebooks, preserved as historical evidence |
| `notebooks/01_*.ipynb`–`09_*.ipynb` | Post-submission guided notebooks; several stages are interface templates rather than end-to-end reproductions |
| `src/supgml/` | Post-submission package extraction of reusable FEM, graph, model, and training mechanics |
| `experiments/` | Configurations for the post-submission runners |
| `gallery/` | Tracked static visualizations of the 102 Chapter 4 cases |
| `docs/assets/thesis-figures/` | Tracked historical figure exports used to explain the reported results |
| `data/` and `runs/` | Ignored local data, checkpoints, and outputs; absent from a fresh clone |
| `docs/` | Maintained MkDocs source |
| `site/` | Ignored generated MkDocs output; do not edit it directly |

The full thesis runs cannot be reproduced from a fresh clone alone because the
FEM datasets, checkpoints, and run summaries are not tracked and no external
data archive is currently documented. The archived submitted notebooks are the
record of the executed historical workflows. The maintained package, guided
notebooks, and experiment runners explain and support continuation of the work;
they should not be mistaken for the exact code path used before submission.

## Install and verify the maintained interfaces

Install DOLFINx, PETSc, and MPI through a supported FEniCSx environment first.
Then install this repository and the relevant optional dependencies into that
environment:

```bash
python -m pip install -e '.[ml,viz,test,docs]'
python -m pytest
supgml-train experiments/ch4_supervised.json --dry-run
```

The dry run validates and displays a configuration; it does not require or
start a training run. See the [installation guide](docs/installation.md) for
platform details and [Continuing this work](docs/continuation.md) for the data
preflight required before any full command.

## Package overview

- `supgml.fem`: finite-element interpolation, assembly, and linear solves.
- `supgml.supg`: SUPG state solves, adjoint objectives, and parameter
  optimization.
- `supgml.benchmarks`: named convection-diffusion-reaction benchmark problems.
- `supgml.stabilization`: standard SUPG, Tabata upwinding, and AFC algorithms.
- `supgml.graph` and `supgml.data`: mesh-to-graph conversion and dataset
  conventions.
- `supgml.models`, `supgml.training`, and `supgml.autograd`: prediction models
  and supervised or FEM-backed training.
- `supgml.optim`: adapters between PyTorch and SciPy optimization.
- `supgml.viz`: optional PyVista and Matplotlib helpers.




## Acknowledgements

### DOLFINx
> I. A. Baratta, J. P. Dean, J. S. Dokken, M. Habera, J. S. Hale, C. N. Richardson, M. E. Rognes, M. W. Scroggs, N. Sime, and G. N. Wells. DOLFINx: The next generation FEniCS problem solving environment, preprint (2023). [[doi.org/10.5281/zenodo.10447666]](https://doi.org/10.5281/zenodo.10447666)

### Basix
Basix is the finite element backend of FEniCSx, responsible for generating finite element basis functions.
> M. W. Scroggs, J. S. Dokken, C. N. Richardson, and G. N. Wells. Construction of arbitrary order finite element degree-of-freedom maps on polygonal and polyhedral cell meshes, ACM Transactions on Mathematical Software 48(2) (2022) 18:1–18:23. [[arΧiv]](https://arxiv.org/abs/2102.11901) [[doi.org/10.1145/3524456]](https://dl.acm.org/doi/10.1145/3524456)

> M. W. Scroggs, I. A. Baratta, C. N. Richardson, and G. N. Wells. Basix: a runtime finite element basis evaluation library, Journal of Open Source Software 7(73) (2022) 3982. [[doi.org/10.21105/joss.03982]](https://joss.theoj.org/papers/10.21105/joss.03982)

### UFL
> M. S. Alnaes, A. Logg, K. B. Ølgaard, M. E. Rognes and G. N. Wells. Unified Form Language: A domain-specific language for weak formulations of partial differential equations, ACM Transactions on Mathematical Software 40 (2014). [[arΧiv]](https://arxiv.org/abs/1211.4047) [[doi.org/10.1145/2566630]](https://dl.acm.org/doi/10.1145/2566630)

### PyTorch
> Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from [http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)

### PyVista
> Sullivan et al., (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). Journal of Open Source Software, 4(37), 1450, [https://doi.org/10.21105/joss.01450](https://joss.theoj.org/papers/10.21105/joss.01450)

### NumPy
> Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: [10.1038/s41586-020-2649-2](https://www.nature.com/articles/s41586-020-2649-2). [(Publisher link)](https://www.nature.com/articles/s41586-020-2649-2).

### Matplotlib
> [J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.](https://ieeexplore.ieee.org/document/4160265)
