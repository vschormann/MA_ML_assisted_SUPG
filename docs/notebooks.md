# Notebook map

The notebooks are experiment records from the thesis. Reusable definitions
have a canonical home in `supgml`; historical copies are retained in notebook
JSON so the submitted analyses remain inspectable.

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

Files containing `copy` are preserved experiment snapshots, not package
sources.

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

When rerunning a notebook, remove its duplicated helper class or function and
replace it with the corresponding package import. Commit each migrated
notebook separately so changes to code cells and regenerated scientific output
remain reviewable.
