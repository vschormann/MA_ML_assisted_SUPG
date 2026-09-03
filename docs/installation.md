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

When using a Conda-forge FEniCSx environment, prefer Conda-forge builds for
packages with compiled runtimes, including PyTorch. Mixing a PyPI PyTorch wheel
with Conda's OpenBLAS and OpenMP libraries can load two copies of
`libomp.dylib` on macOS.

## macOS: kernel dies when importing PyTorch

On Apple Silicon, a Jupyter kernel in a Conda-forge FEniCSx environment may
terminate immediately when `import torch` is the first import. VS Code usually
reports only:

```text
Disposing session as kernel process died ExitCode: undefined, Reason:
```

Running the same import in a terminal can expose the underlying error:

```bash
python -c "import torch"
```

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

This happens when the environment contains both Conda's `libomp.dylib` and the
copy bundled in a PyPI PyTorch wheel. Importing NumPy before PyTorch may avoid
the abort temporarily:

```python
import numpy
import torch
```

The durable fix is to use the Conda-forge PyTorch build so the numerical stack
shares one OpenMP runtime. The following versions are validated with the
Python 3.13, DOLFINx 0.9 environment used by this project:

```bash
conda activate fenicsx-env
python -m pip uninstall -y torch torchvision
conda install -c conda-forge \
  "pytorch=2.7.1" "torchvision=0.22.1" \
  --freeze-installed
```

`--freeze-installed` prevents Conda from unnecessarily downgrading DOLFINx,
PETSc, MPI, and the compiler stack while solving the PyTorch dependencies.
Check the proposed transaction before accepting it, particularly when using
different FEniCSx or Python versions.

Do not set `KMP_DUPLICATE_LIB_OK=TRUE` as a permanent workaround. It suppresses
the safety check while retaining multiple OpenMP runtimes, which can cause
crashes or incorrect numerical results.

Restart VS Code's notebook kernel after changing the environment, select
`fenicsx-env`, and verify the formerly failing import order:

```bash
python -c "import torch; import numpy; import dolfinx; print(torch.__version__, dolfinx.__version__)"
```

## Verify

```bash
python -c "import supgml; print(supgml.__version__)"
python -m pytest
```
