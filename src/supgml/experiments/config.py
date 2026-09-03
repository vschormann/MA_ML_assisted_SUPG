"""JSON experiment configuration with lightweight validation."""

import json
from pathlib import Path


REQUIRED_TRAINING_KEYS = {
    "name",
    "dataset",
    "architectures",
    "mode",
    "epochs",
    "batch_size",
    "learning_rate",
    "output_dir",
}


def project_root(start=None):
    """Find the nearest parent containing this project's ``pyproject.toml``."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        metadata = candidate / "pyproject.toml"
        if metadata.exists() and 'name = "supgml"' in metadata.read_text(encoding="utf-8"):
            return candidate
    raise FileNotFoundError("could not locate the supgml repository root")


def load_config(path):
    """Load and validate an experiment JSON file."""

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    missing = REQUIRED_TRAINING_KEYS - set(config)
    if missing:
        raise ValueError("missing experiment keys: {}".format(", ".join(sorted(missing))))
    if config["mode"] not in {"supervised", "self_supervised"}:
        raise ValueError("mode must be 'supervised' or 'self_supervised'")
    if not config["architectures"]:
        raise ValueError("architectures cannot be empty")
    config["config_path"] = str(path)
    return config
