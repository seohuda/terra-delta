"""Offline seasonal appearance and same-geometry consistency contracts."""
import numpy as np
import pytest
import torch
from PIL import Image

from terradelta.data.dataset_v2 import MEAN, STD, IndependentChangeDataset, V2Transform, _tensor
from terradelta.data.season_v21 import (
    MAX_RGB_DELTA,
    SeasonTransform,
    photometric_view,
    photometric_view_numpy,
)


def scene():
    pre = np.full((29, 37, 3), 12, np.uint8)
    post = pre.copy()
    masks = np.zeros((2, 29, 37), bool)
    masks[0, 4:12, 7:19] = True
    masks[0, 15:24, 25] = True  # Include a one-pixel structure.
    masks[1, 18:26, 3:12] = True
    masks[1, 9, 8:14] = True  # Independent labels may overlap.
    pre[masks[1]] = [15, 235, 15]
    post[masks[0]] = 240
    valid = np.ones(pre.shape[:2], bool)
    valid[:3, :9] = False
    return pre, post, masks, valid


def normalized(pre, post, batch=1):
    return torch.cat([_tensor(pre), _tensor(post)])[None].repeat(batch, 1, 1, 1)


def rgb(image):
    mean = image.new_tensor(np.tile(MEAN, 2))[None, :, None, None]
    std = image.new_tensor(np.tile(STD, 2))[None, :, None, None]
    return image * std + mean


@pytest.mark.parametrize("registration", [0., 1.])
def test_transform_deterministic_v2_geometry_and_class_masks(registration):
    inputs = scene()
    saved = [value.copy() for value in inputs]
    transform = SeasonTransform(relative_probability=registration)
    geometry = V2Transform(relative_probability=registration, max_translation=3.,
                           max_rotation=.8, max_scale=.01, photometric=False)
    first = transform(*inputs, seed=73021)
    second = transform(*inputs, seed=73021)
    expected = geometry(*inputs, seed=73021)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert left.flags.c_contiguous
    for output, reference in zip(first[2:], expected[2:]):
        np.testing.assert_array_equal(output, reference)
    np.testing.assert_array_equal(first[2].sum((1, 2)), inputs[2].sum((1, 2)))
    assert (first[2][0] & first[2][1]).sum() == (inputs[2][0] & inputs[2][1]).sum()
    for original, snapshot in zip(inputs, saved):
        np.testing.assert_array_equal(original, snapshot)
    for output, reference in zip(first[:2], expected[:2]):
        assert output.dtype == np.uint8
        assert np.abs(output.astype(float) - reference).max() <= MAX_RGB_DELTA * 255 + .5
        assert np.any(output != reference)


def test_registration_moves_only_pre_and_zero_appearance_matches_v2():
    inputs = scene()
    expected = V2Transform(relative_probability=1., max_translation=3., max_rotation=.8,
                           max_scale=.01, photometric=False)(*inputs, seed=9)
    unregistered = V2Transform(relative_probability=0, photometric=False)(*inputs, seed=9)
    for options in ({"strength": 0}, {"photometric": False}):
        output = SeasonTransform(relative_probability=1, **options)(*inputs, seed=9)
        for left, right in zip(output, expected):
            np.testing.assert_array_equal(left, right)
        assert not np.array_equal(output[0], unregistered[0])
        for left, right in zip(output[1:], unregistered[1:]):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("strength", [.1, .5, 1.])
def test_bounded_season_preserves_positive_locations_and_empty_negatives(strength):
    inputs = scene()
    transform = SeasonTransform(relative_probability=0, strength=strength)
    geometry = V2Transform(relative_probability=0, photometric=False)
    for seed in range(16):
        pre, post, masks, valid = transform(*inputs, seed=seed)
        expected = geometry(*inputs, seed=seed)
        # Threshold margins remain large enough even for the one-pixel building.
        np.testing.assert_array_equal(post[..., 0] > 127, masks[0])
        np.testing.assert_array_equal(pre[..., 1] > 127, masks[1])
        for output, reference in zip((pre, post), expected[:2]):
            assert np.abs(output.astype(float) - reference).max() <= MAX_RGB_DELTA * strength * 255 + .5
        empty = np.zeros_like(inputs[2])
        negative = transform(inputs[0], inputs[1], empty, inputs[3], seed=seed)
        assert not negative[2].any()
        np.testing.assert_array_equal(negative[3], valid)


def test_dataset_adapter_preserves_absent_targets_presence_and_review_scope(tmp_path):
    pre, post, masks, valid = scene()
    for name, value in (("pre", pre), ("post", post), ("building", masks[0].astype(np.uint8) * 255),
                        ("valid", valid.astype(np.uint8) * 255)):
        Image.fromarray(value).save(tmp_path / f"{name}.png")
    row = {"id": "positive", "pre": str(tmp_path / "pre.png"), "post": str(tmp_path / "post.png"),
           "new_building": str(tmp_path / "building.png"), "tree_removal": "absent",
           "review_mask": str(tmp_path / "valid.png")}
    negative = {**row, "id": "negative", "new_building": "absent"}
    transform = SeasonTransform()
    dataset = IndependentChangeDataset([row, negative], transform=transform)
    sample = dataset.get(0, seed=17)
    target = masks.copy()
    target[1] = False
    expected = transform(pre, post, target, valid, seed=17)
    torch.testing.assert_close(sample["target"], torch.from_numpy(expected[2].astype(np.float32)))
    torch.testing.assert_close(sample["valid_mask"], torch.from_numpy(expected[3]))
    assert sample["presence"].tolist() == [1., 0.]
    assert sample["presence_valid"].tolist() == [True, True]
    empty = dataset.get(1, seed=17)
    assert not empty["target"].any()
    assert empty["presence"].tolist() == [0., 0.]
    assert empty["presence_valid"].tolist() == [True, True]
    assert torch.isfinite(sample["image"]).all()


def test_view_repeatable_independent_and_does_not_consume_global_rng():
    pre, post, _, _ = scene()
    image = normalized(pre, pre, batch=2)
    saved = image.clone()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    first = photometric_view(image, 20261004)
    torch.testing.assert_close(first, photometric_view(image, 20261004), rtol=0, atol=0)
    assert not torch.equal(first, photometric_view(image, 20261005))
    assert not torch.equal(first[:, :3], first[:, 3:])
    assert not torch.equal(first[0], first[1])
    torch.testing.assert_close(image, saved, rtol=0, atol=0)
    current = np.random.get_state()
    assert current[0] == numpy_state[0] and current[2:] == numpy_state[2:]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert torch.equal(torch_state, torch.random.get_rng_state())
    # The CPU adapter uses the very same transform and normalization.
    pre_view, post_view = photometric_view_numpy(pre, post, 31)
    expected = rgb(photometric_view(normalized(pre, post), 31))[0]
    np.testing.assert_array_equal(pre_view, (expected[:3] * 255).round().byte().permute(1, 2, 0).numpy())
    np.testing.assert_array_equal(post_view, (expected[3:] * 255).round().byte().permute(1, 2, 0).numpy())


def test_view_has_no_geometry_displacement_and_preserves_structure():
    pre, post, masks, _ = scene()
    image = normalized(pre, post)
    for seed in range(24):
        view = photometric_view(image, seed)
        pixels = rgb(view)
        np.testing.assert_array_equal((pixels[0, 1] > .5).numpy(), masks[1])
        np.testing.assert_array_equal((pixels[0, 3] > .5).numpy(), masks[0])
        assert (pixels - rgb(image)).abs().max() <= MAX_RGB_DELTA + 1e-6
        assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (8, 1), (7, 11)])
def test_view_finite_float32_bounds_and_zero_strength(dtype, shape):
    pre = np.zeros((*shape, 3), np.uint8)
    post = np.full_like(pre, 255)
    image = normalized(pre, post).to(dtype)
    view = photometric_view(image, np.int64(-7))
    assert view.shape == image.shape and view.device == image.device
    assert view.dtype == torch.float32 and torch.isfinite(view).all()
    assert rgb(view).min() >= -1e-6 and rgb(view).max() <= 1 + 1e-6
    middle = normalized(np.full_like(pre, 120), np.full_like(post, 150)).to(dtype)
    torch.testing.assert_close(photometric_view(middle, 7, strength=0), middle.float(), rtol=0, atol=0)
    np.testing.assert_array_equal(photometric_view_numpy(pre, post, 7, strength=0)[0], pre)


def test_view_keeps_autograd_finite():
    pre, post, _, _ = scene()
    image = normalized(pre, post).requires_grad_()
    photometric_view(image, 17).square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_view_preserves_cuda_device():
    pre, post, _, _ = scene()
    image = normalized(pre, post).cuda()
    first = photometric_view(image, 7)
    assert first.device == image.device and first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, photometric_view(image, 7), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"strength": -.1}, {"strength": 1.1}, {"strength": float("nan")},
    {"relative_probability": 1.1}, {"relative_probability": -1},
    {"max_translation": -1}, {"max_translation": 6.1}, {"max_translation": float("inf")},
    {"max_rotation": 1.6}, {"max_scale": .03}, {"photometric": "false"},
])
def test_transform_validates_settings(kwargs):
    with pytest.raises(ValueError):
        SeasonTransform(**kwargs)


@pytest.mark.parametrize("image", [
    torch.zeros(6, 4, 4), torch.zeros(1, 3, 4, 4), torch.zeros(0, 6, 4, 4),
    torch.zeros(1, 6, 0, 4), torch.zeros(1, 6, 4, 4, dtype=torch.uint8),
    torch.full((1, 6, 4, 4), float("nan")), torch.full((1, 6, 4, 4), float("inf")),
    torch.full((1, 6, 4, 4), 1e100, dtype=torch.float64), np.zeros((1, 6, 4, 4)),
])
def test_view_rejects_invalid_input(image):
    with pytest.raises(ValueError):
        photometric_view(image, 7)


def test_validates_seeds_strength_and_pair_shapes():
    pre, post, masks, valid = scene()
    for seed in (1.5, "7", None):
        with pytest.raises(ValueError, match="seed"):
            photometric_view(normalized(pre, post), seed)
        with pytest.raises(ValueError, match="seed"):
            SeasonTransform()(pre, post, masks, valid, seed)
    for strength in (-1, 1.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="strength"):
            photometric_view(normalized(pre, post), 7, strength=strength)
    with pytest.raises(ValueError, match="pre/post"):
        photometric_view_numpy(pre, post[:-1], 7)
    with pytest.raises(ValueError, match="pre/post"):
        SeasonTransform()(pre.astype(float), post, masks, valid, 7)
    with pytest.raises(ValueError, match="masks"):
        SeasonTransform()(pre, post, masks[:1], valid, 7)
