import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_all_notebooks_are_valid_json():
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    assert len(notebooks) == 41
    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["nbformat"] == 4
        assert isinstance(document["cells"], list)


def test_canonical_notebooks_declare_status():
    canonical = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(canonical) == 9
    for path in canonical:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["metadata"]["supgml"]["status"] == "canonical"
