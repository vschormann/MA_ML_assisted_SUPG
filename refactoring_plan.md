# Refactoring plan and record

The submitted repository is permanently identified by commit
`aed55ecdaf7c99b4f0f89662e48eab106de8013f`. This file records the
post-submission readability refactor.

## Completed stages

- [x] Add `pyproject.toml`, optional dependency groups, and documentation.
- [x] Separate FEM utilities, SUPG solvers, benchmarks, and visualization.
- [x] Extract standard SUPG, Tabata, Kuzmin AFC, and BJK AFC implementations
  from `revised_approximations.ipynb`.
- [x] Separate mesh-to-graph conversion, schema, persistence, and datasets.
- [x] Add configurable graph-model and output-constraint APIs.
- [x] Separate supervised training, differentiable FEM loss, and SciPy adapters.
- [x] Keep top-level compatibility imports for the submitted notebooks.
- [ ] Validate DOLFINx integration in the supported FEniCSx environment.
- [x] Preserve all submitted notebooks in categorized archive directories.
- [x] Add nine ordered canonical notebooks aligned with Chapters 4 and 5.
- [x] Consolidate Chapter 4 training into supervised and self-supervised
  configuration matrices.
- [x] Extract the revised width-256 MLP/GATv2 architectures and Chapter 5 loss.
- [ ] Execute the canonical long-running experiments in the DOLFINx environment.

## Notebook policy

The submitted notebooks are retained under `notebooks/archive` as research
records, including historical outputs and local definitions. They are not
rewritten. The package implementation and the ordered notebooks directly under
`notebooks/` are canonical after this refactor.

## Commit structure

The refactor is divided into reviewable commits for:

1. packaging and documentation;
2. numerical solvers and benchmark separation;
3. graph learning and training APIs; and
4. tests, compatibility checks, and final migration notes.
