import json
from pathlib import Path

import pytest

from supgml.experiments import load_config, project_root


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name",
    ["ch4_supervised.json", "ch4_self_supervised.json", "ch5_revised.json"],
)
def test_experiment_configs_are_valid(name):
    config = load_config(ROOT / "experiments" / name)
    assert config["architectures"]
    assert config["epochs"] > 0


def test_missing_config_fields_are_reported(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"name": "invalid"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing experiment keys"):
        load_config(path)


def test_project_root_is_found_from_notebook_directory():
    assert project_root(ROOT / "notebooks") == ROOT
