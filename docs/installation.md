# Installation

## Lightweight installation

The base package contains dependency-light schemas and numerical utilities:

```bash
python -m pip install -e .
```

Install optional features only when needed:

```bash
python -m pip install -e '.[ml]'
python -m pip install -e '.[viz]'
python -m pip install -e '.[ml,viz,test]'
```

## FEniCSx environment

DOLFINx depends on MPI, PETSc, and compiled finite-element libraries. Create a
working DOLFINx environment using an official FEniCSx binary, Conda
environment, or container first. Then install this repository into that
environment:

```bash
python -m pip install -e '.[fem,ml,viz]'
```

`dolfinx` itself is deliberately not declared as an ordinary PyPI dependency:
its installation method and native dependencies depend on the platform.

## Verify

```bash
python -c "import supgml; print(supgml.__version__)"
python -m pytest
```
