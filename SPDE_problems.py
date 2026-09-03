"""Compatibility imports for the original benchmark module."""

from supgml.benchmarks import *  # noqa: F401,F403
from supgml.benchmarks import __all__ as _benchmark_exports
from supgml.data.repository import Data_to_solver

__all__ = list(_benchmark_exports) + ["Data_to_solver"]
