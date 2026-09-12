# Guided thesis workflow

These nine maintained notebooks follow the argument of the submitted thesis.
Notebooks 01-05 cover the foundations and heterogeneous Chapter 4 study;
notebooks 06-09 cover the revised AFC-BJK study in Chapter 5. They were created
after submission to explain the refactored `supgml` interfaces.

They are a reading and continuation path, not a claim of one-click
reproducibility. Some cells are runnable examples, while several later stages
show the intended interface with commented placeholders or require ignored
datasets, checkpoints, and run summaries. The submitted notebooks and their
historical outputs remain under `archive/`; read `../docs/continuation.md` for
the complete provenance and artefact boundary.

| Order | Notebook | Scientific role | Execution scope |
| --- | --- | --- | --- |
| 01 | `01_supg_objectives.ipynb` | SUPG parameters, objectives, and direct optimization | Runnable foundation example in a DOLFINx environment |
| 02 | `02_ch4_generate_dataset.ipynb` | Seven benchmark families and the heterogeneous dataset | Documents the matrix/schema; expensive generation calls are sketched |
| 03 | `03_ch4_train_supervised.ipynb` | Supervised architecture comparison | Loads the refactored config; full CLI run needs the ignored Chapter 4 data |
| 04 | `04_ch4_train_self_supervised.ipynb` | Adjoint-backed architecture comparison | Shows the autograd bridge and config; full run needs data and FEM solvers |
| 05 | `05_ch4_evaluate_models.ipynb` | Chapter 4 evidence, limitations, and conclusion | Analysis template; historical plots and outputs are in the archive/gallery |
| 06 | `06_ch5_build_afc_target.ipynb` | AFC-BJK reference and revised target | Runs the SPDE setup; the AFC target/optimization step is an explicit placeholder |
| 07 | `07_ch5_train_revised_models.ipynb` | Wide revised MLP and GATv2 | Loads the refactored config; full run needs the ignored revised case |
| 08 | `08_ch5_analyze_revised_models.ipynb` | Perturbations, lower bound, and Figures 38–42 | Requires an ignored `runs/ch5_revised/summary.json` or adaptation to historical files |
| 09 | `09_render_thesis_figures.ipynb` | Figure-provenance reporting | Reporting template; the figure specification is intentionally unpopulated |

## Validate configurations

From the repository root, a dry run validates and displays a configuration
without loading data or starting training:

```bash
supgml-train experiments/ch4_supervised.json --dry-run
supgml-train experiments/ch4_self_supervised.json --dry-run
supgml-train experiments/ch5_revised.json --dry-run
```

## Start a new refactored run

Only after restoring or regenerating the inputs named by each configuration,
use:

```bash
supgml-train experiments/ch4_supervised.json
supgml-train experiments/ch4_self_supervised.json
supgml-train experiments/ch5_revised.json
```

These are post-submission runners, not the exact processes that produced the
submitted checkpoints. They write new outputs under the ignored `runs/`
directory. The Chapter 4 configurations request 20,000 epochs; Chapter 5
requests 200,000 MLP epochs and 150,000 GATv2 epochs. Read the continuation
guide before launching them.

The two chapters remain separate because Chapter 5 changes the reference
solution, objective, model scope, model capacity, loss, and optimization loop.
The Chapter 4 approach is the submitted thesis evidence rather than an obsolete
implementation to overwrite.

The original notebooks and their outputs are under `archive/`. See
`archive/README.md` before using them.
