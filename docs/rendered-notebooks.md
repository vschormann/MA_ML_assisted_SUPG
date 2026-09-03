# Rendered canonical notebooks

These are static HTML exports of the nine canonical notebooks. They open in
Safari or any other browser without Jupyter, Python, or a running kernel.
They are documentation snapshots: code is shown but not executed in the
browser. Use the corresponding `.ipynb` file when you want to run or modify an
experiment.

| Notebook | Browser version | Purpose |
| --- | --- | --- |
| 01 | [SUPG objectives](rendered-notebooks/01_supg_objectives.html) | FEM setup, SUPG form, and discrete adjoint |
| 02 | [Chapter 4 data generation](rendered-notebooks/02_ch4_generate_dataset.html) | FEM cases, graph schema, and validation |
| 03 | [Chapter 4 supervised training](rendered-notebooks/03_ch4_train_supervised.html) | Common architecture-comparison task |
| 04 | [Chapter 4 FEM-backed training](rendered-notebooks/04_ch4_train_self_supervised.html) | PyTorch/FEniCSx adjoint bridge |
| 05 | [Chapter 4 evaluation](rendered-notebooks/05_ch4_evaluate_models.html) | Parameter and FEM-solution comparison |
| 06 | [Chapter 5 AFC-BJK target](rendered-notebooks/06_ch5_build_afc_target.html) | Revised SPDE and reference solution |
| 07 | [Chapter 5 revised training](rendered-notebooks/07_ch5_train_revised_models.html) | Revised MLP/GATv2 selection |
| 08 | [Chapter 5 analysis](rendered-notebooks/08_ch5_analyze_revised_models.html) | Target ambiguity and model interpretation |
| 09 | [Thesis figure rendering](rendered-notebooks/09_render_thesis_figures.html) | Read-only figure provenance |

To regenerate the exports after editing a canonical notebook, run this command
from the repository root:

```bash
python -m jupyter nbconvert --to html --output-dir docs/rendered-notebooks \
  notebooks/*.ipynb
```

The archive is intentionally not exported: it preserves submitted and
exploratory evidence, whereas this page is a browser-readable guide to the
canonical workflow.
