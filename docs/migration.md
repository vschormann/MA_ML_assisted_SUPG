# Migration from top-level modules

The original imports continue to work during the notebook migration. New code
should use the package paths below.

| Previous import | New import |
| --- | --- |
| `FEniCSx_solver` | `supgml.fem`, `supgml.supg`, `supgml.viz` |
| `SPDE_problems` | `supgml.benchmarks`, `supgml.data` |
| `Training_utils` | `supgml.data`, `supgml.training` |
| `SUPG_prediction_models` | `supgml.models` |
| `FEniCSx_PyTorch_interface` | `supgml.autograd`, `supgml.training` |
| `PyTorch_SciPy_interface` | `supgml.optim` |

Notebook-local helpers move as follows:

- `interpolate_expr`, `curve_plotter`: `supgml.fem`;
- `int_to_prblm`: `supgml.benchmarks.create`;
- `fs_to_edge_index`, `fs_to_x`, `relative_position`: `supgml.graph`;
- `save_input_data`, `save_target_values`: `supgml.data`;
- `yh_std`, `tabata`, `F_AFC_Kuzmin`, `F_AFC_BJK`:
  `supgml.stabilization`;
- notebook model classes: `supgml.models.create_model`;
- duplicated training and batched FEM loss functions: `supgml.training`.
