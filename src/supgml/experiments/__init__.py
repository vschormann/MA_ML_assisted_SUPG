"""Configuration-driven experiment entry points used by canonical notebooks."""

from .config import load_config, project_root


def run_training(config):
    """Lazily load the ML experiment runner."""

    from .training import run_training as run

    return run(config)

__all__ = ["load_config", "project_root", "run_training"]
