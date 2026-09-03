# Notebook map

Nine ordered notebooks under `notebooks/` now follow the thesis argument.
Reusable definitions have a canonical home in `supgml`; all 32 submitted
notebooks and their outputs are retained under `notebooks/archive/`.

The canonical sequence is:

1. SUPG objectives and direct optimization;
2. Chapter 4 dataset generation;
3. Chapter 4 supervised architecture comparison;
4. Chapter 4 self-supervised architecture comparison;
5. Chapter 4 evaluation;
6. Chapter 5 AFC-BJK target construction;
7. Chapter 5 revised MLP/GATv2 training;
8. Chapter 5 revised-model analysis; and
9. deterministic thesis-figure rendering.

## Data preparation

| Notebook | Responsibility | Canonical package API |
| --- | --- | --- |
| `data_generation.ipynb` | base mesh graphs and targets | `supgml.graph.GraphBuilder`, `supgml.data.CaseRepository` |
| `edge_attr_set_creation.ipynb` | geometric and sensitivity edges | `supgml.graph.relative_position`, `finite_difference_sensitivity` |
| `redo_graphs.ipynb` | graph-schema migration | `supgml.graph`, `supgml.data` |

## Numerical methods

| Notebook | Responsibility | Canonical package API |
| --- | --- | --- |
| `analysis_optimal_parameters.ipynb` | direct SUPG-parameter optimization | `supgml.supg`, `supgml.benchmarks` |
| `revised_approximations.ipynb` | Tabata and AFC comparisons | `supgml.stabilization` |
| `activation_plot.ipynb` | saturating loss inspection | `supgml.supg.SaturatingLoss` |

## Supervised model experiments

`Train_MLP`, `Train_GCN`, `Train_SAGE`, `Train_GAT`, `Train_GATv2`,
`Train_Attention`, `Train_MHA`, `Train_edge_attr`, and `Train_globalizer`
compare architectures and graph features. Their shared implementation is now:

- `supgml.models.create_model` for MLP, GCN, GraphSAGE, GAT, and GATv2;
- `supgml.models.BoundedOutput` for admissible SUPG parameters; and
- `supgml.training.train_epoch` or `fit` for training.

Files containing `copy` are preserved in the archive as experiment snapshots,
not package sources. In particular, `Train_revised copy.ipynb` contains the
revised GATv2 experiment.

## FEM-backed training

The notebooks whose names end in `_self_supervised`, together with
`self_supervised_training.ipynb`, differentiate the FEM objective through an
adjoint gradient. The canonical implementation is in `supgml.autograd` and
`supgml.training.self_supervised_train`.

## Evaluation and publication

| Notebook | Responsibility | Canonical package API |
| --- | --- | --- |
| `Test_set_analysis.ipynb` | model/FEM comparison | `supgml.evaluation.evaluate_models` |
| `revised_data_analysis.ipynb` | revised feature/model analysis | `supgml.evaluation`, `supgml.viz` |
| `gallery_creator.ipynb` | publication images and gallery | `supgml.viz` |
| `Train_revised*.ipynb` | revised-data training experiments | `supgml.models`, `supgml.training` |

## Migration rule

Do not edit an archived submitted notebook in place. Extend a canonical
notebook or create a new configuration-driven experiment, keeping regenerated
scientific output separate from the archived evidence.
