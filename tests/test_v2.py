import numpy as np
import torch

from terradelta.data.dataset_v2 import V2Transform
from terradelta.models.siamese_v2 import TerraDeltaSiameseV2
from terradelta.training.losses_v2 import V2Loss


def test_v2_model_has_shared_encoder_independent_outputs():
    model = TerraDeltaSiameseV2(alignment=True, alignment_scales=(3, 4))
    x = torch.randn(2, 6, 64, 64)
    with torch.inference_mode():
        out = model(x)
    assert out["segmentation"].shape == (2, 2, 64, 64)
    assert out["presence"].shape == (2, 2)
    assert model.encoder is model.encoder
    assert not hasattr(model, "pre_encoder")
    assert not hasattr(model, "post_encoder")


def test_v2_loss_respects_pixel_and_presence_validity():
    loss_fn = V2Loss()
    outputs = {
        "segmentation": torch.zeros(1, 2, 8, 8, requires_grad=True),
        "presence": torch.zeros(1, 2, requires_grad=True),
    }
    target = torch.zeros(1, 2, 8, 8)
    target[:, 0, 2:5, 2:5] = 1
    valid = torch.zeros(1, 8, 8, dtype=torch.bool)
    valid[:, 2:5, 2:5] = True
    presence = torch.tensor([[1.0, 0.0]])
    presence_valid = torch.tensor([[True, False]])
    loss, parts = loss_fn(outputs, target, valid, presence, presence_valid)
    assert torch.isfinite(loss)
    assert set(parts) == {"bce", "dice_loss", "presence_loss"}
    loss.backward()
    assert outputs["segmentation"].grad is not None
    assert outputs["presence"].grad is not None


def test_relative_transform_is_deterministic_and_does_not_warp_labels():
    pre = np.full((32, 32, 3), 80, np.uint8)
    post = np.full((32, 32, 3), 120, np.uint8)
    masks = np.zeros((2, 32, 32), bool)
    masks[0, 8:16, 10:18] = True
    valid = np.ones((32, 32), bool)
    transform = V2Transform(
        relative_probability=1.0,
        max_translation=5,
        max_rotation=1,
        max_scale=.01,
        photometric=False,
    )
    a = transform(pre, post, masks, valid, seed=7)
    b = transform(pre, post, masks, valid, seed=7)
    for left, right in zip(a, b):
        assert np.array_equal(left, right)
    assert a[2].sum() == masks.sum()
    assert a[3].sum() == valid.sum()
