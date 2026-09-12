# Canonical notebook sequence

These nine notebooks follow the argument of the submitted thesis. Notebooks
01-05 cover the foundations and heterogeneous Chapter 4 study; notebooks 06-09
cover the revised AFC-BJK study in Chapter 5. All are small clients of the
installable `supgml` package; reusable definitions do not belong in notebook
cells.

| Order | Notebook | Role |
| --- | --- | --- |
| 01 | `01_supg_objectives.ipynb` | SUPG parameters, objectives, and direct optimization |
| 02 | `02_ch4_generate_dataset.ipynb` | **Thesis:** seven benchmark families and heterogeneous dataset |
| 03 | `03_ch4_train_supervised.ipynb` | **Thesis:** supervised architecture comparison |
| 04 | `04_ch4_train_self_supervised.ipynb` | **Thesis:** adjoint-backed architecture comparison |
| 05 | `05_ch4_evaluate_models.ipynb` | **Thesis:** Chapter 4 evidence, limitations, and conclusion |
| 06 | `06_ch5_build_afc_target.ipynb` | **Chapter 5:** AFC-BJK reference and revised target |
| 07 | `07_ch5_train_revised_models.ipynb` | **Chapter 5:** wide revised MLP and GATv2 |
| 08 | `08_ch5_analyze_revised_models.ipynb` | **Chapter 5:** perturbations, lower bound, and Figures 38–42 |
| 09 | `09_render_thesis_figures.ipynb` | Deterministic thesis-figure reporting |

Training is started from the repository root:

```bash
supgml-train experiments/ch4_supervised.json --dry-run
supgml-train experiments/ch4_supervised.json
supgml-train experiments/ch4_self_supervised.json
supgml-train experiments/ch5_revised.json
```

The two chapters remain separate because Chapter 5 changes the reference
solution, objective, model scope, model capacity, loss, and optimization loop.
The Chapter 4 approach is the submitted thesis evidence rather than an obsolete
implementation to overwrite.

The original notebooks and their outputs are under `archive/`. See
`archive/README.md` before using them.
