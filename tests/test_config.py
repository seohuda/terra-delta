from pathlib import Path
import pytest
from terradelta.utils.io import load_config


def test_all_presets_load():
    for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"):
        assert isinstance(load_config(path), dict)


def test_reject_nonmapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- wrong\n")
    with pytest.raises(ValueError):
        load_config(path)
