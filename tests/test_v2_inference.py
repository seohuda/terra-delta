import numpy as np
import pytest
import torch
from terradelta.inference.v2 import V2Predictor, output_row
from terradelta.models.siamese_v2 import load_v2_checkpoint, save_v2_checkpoint


class IndependentModel(torch.nn.Module):
    def forward(self, image):
        return {
            "segmentation": torch.full((len(image), 2, 256, 256), 4.0),
            "presence": torch.tensor([-4.0, 4.0]).expand(len(image), 2),
        }


def test_independent_overlap_and_presence_precedes_zero_pixel_threshold():
    config = {"postprocess": {"mode": "independent", "pixel_threshold": 0.0, "presence_threshold": 0.5}}
    p, q = V2Predictor(None, config, model=IndependentModel()).probabilities(torch.zeros(1, 6, 256, 256))
    row = output_row("001", p[0], q[0], config)
    assert row["new_building"] == "" and row["tree_removal"]
    config["postprocess"]["presence_threshold"] = None
    row = output_row("001", p[0], q[0], config)
    assert row["new_building"] == row["tree_removal"] and row["new_building"]


def test_empty_outputs_and_nan_rejected():
    p = np.zeros((2, 256, 256), np.float32)
    row = output_row("001", p, np.zeros(2), {"postprocess": {"presence_threshold": None}})
    assert row == {"id": "001", "new_building": "", "tree_removal": ""}
    p[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        output_row("001", p, np.zeros(2), {})


def test_checkpoint_rejects_class_order(tmp_path):
    path = tmp_path / "model.pt"
    model = torch.nn.Conv2d(3, 2, 1)
    save_v2_checkpoint(path, model)
    load_v2_checkpoint(path, model)
    state = torch.load(path, weights_only=True)
    state["classes"] = ["tree_removal", "new_building"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="class order"):
        load_v2_checkpoint(path, model)
