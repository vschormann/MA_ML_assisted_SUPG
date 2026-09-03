# Refactoring process

Commit `aed55ecdaf7c99b4f0f89662e48eab106de8013f` is the immutable reference for
the repository as submitted with the thesis. All commits after it reorganize,
document, test, or clarify the implementation.

The migration is intentionally staged:

1. add packaging metadata and document the submitted baseline;
2. extract numerical solvers, benchmark definitions, and visualization;
3. extract graph construction, datasets, models, and training interfaces;
4. retain top-level compatibility modules for the existing notebooks;
5. validate imports, pure numerical functions, model shapes, and FEM gradients;
6. gradually reduce notebooks to configuration, package calls, and analysis.

The reduction uses an archive-and-replace strategy: all submitted notebooks
remain intact under `notebooks/archive`, while nine clean notebooks present the
workflow in thesis order. Architecture sweeps are represented by JSON
configuration rather than copied notebooks.

Behavior-changing scientific experiments should be clearly identified in
future commits and should not be described as part of this readability-only
refactoring series.
