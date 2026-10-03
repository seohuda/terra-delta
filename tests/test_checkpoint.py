import pytest
import torch

from terradelta.models.checkpoint import load_checkpoint, save_checkpoint
from terradelta.models.factory import build_model


def test_official_metadata(official_checkpoint):
    metadata = load_checkpoint(official_checkpoint, build_model())
    assert metadata == {"classes": ["background", "new_building", "tree_removal"],
                        "in_channels": 6, "encoder": "resnet18", "steps": 100, "seed": 0}


def test_plain_and_wrapped_roundtrip(tmp_path):
    # A tiny state tests format support without any training or a large disk file.
    model = torch.nn.Conv2d(6, 3, 1)
    wrapped = tmp_path / "wrapped.pt"
    save_checkpoint(wrapped, model, steps=25, seed=0)
    copy = torch.nn.Conv2d(6, 3, 1)
    assert load_checkpoint(wrapped, copy)["steps"] == 25
    for a, b in zip(model.parameters(), copy.parameters()):
        torch.testing.assert_close(a, b)
    plain = tmp_path / "plain.pt"
    torch.save(model.state_dict(), plain)
    assert load_checkpoint(plain, copy) == {}


def test_checkpoint_rejects_metadata_mismatch(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"state_dict": {"x": torch.zeros(1)}, "classes": ["tree_removal", "background", "new_building"]}, p)
    with pytest.raises(ValueError, match="class order"):
        load_checkpoint(p)
