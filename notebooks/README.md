# Canonical notebook sequence

These notebooks follow the argument of the submitted thesis. They are small
clients of the installable `supgml` package; reusable definitions do not belong
in notebook cells.

| Order | Notebook | Thesis role |
| --- | --- | --- |
| 01 | `01_supg_objectives.ipynb` | SUPG parameters, objectives, and direct optimization |
| 02 | `02_ch4_generate_dataset.ipynb` | Original heterogeneous dataset |
| 03 | `03_ch4_train_supervised.ipynb` | Supervised architecture matrix |
| 04 | `04_ch4_train_self_supervised.ipynb` | Adjoint-backed architecture matrix |
| 05 | `05_ch4_evaluate_models.ipynb` | Chapter 4 comparisons and discussion |
| 06 | `06_ch5_build_afc_target.ipynb` | AFC-BJK reference and revised target |
| 07 | `07_ch5_train_revised_models.ipynb` | Wide revised MLP and GATv2 |
| 08 | `08_ch5_analyze_revised_models.ipynb` | Perturbations, lower bounds, and Figures 38-42 |
| 09 | `09_render_thesis_figures.ipynb` | Deterministic reporting only |

Training is started from the repository root:

```bash
supgml-train experiments/ch4_supervised.json --dry-run
supgml-train experiments/ch4_supervised.json
supgml-train experiments/ch4_self_supervised.json
supgml-train experiments/ch5_revised.json
```

Chapter 4 and Chapter 5 remain separate because Chapter 5 changes the reference
solution, objective, model scope, model capacity, loss, and optimization loop.
The earlier approach is part of the thesis evidence rather than an obsolete
implementation to overwrite.

The original notebooks and their outputs are under `archive/`. See
`archive/README.md` before using them.
