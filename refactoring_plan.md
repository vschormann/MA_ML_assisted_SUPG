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
- [ ] Migrate experiment notebooks individually when they are next rerun.

## Notebook policy

The notebooks are retained as research records, including their historical
outputs and local class definitions. The package implementation is canonical
after this refactor. When an experiment is rerun, replace its local helper
definitions with imports documented in `docs/migration.md`; avoid rewriting all
notebook JSON at once because that would obscure the scientific history.

## Commit structure

The refactor is divided into reviewable commits for:

1. packaging and documentation;
2. numerical solvers and benchmark separation;
3. graph learning and training APIs; and
4. tests, compatibility checks, and final migration notes.
