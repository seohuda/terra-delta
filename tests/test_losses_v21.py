import math

import pytest
import torch

from terradelta.training.losses_v2 import V2Loss
from terradelta.training.losses_v21 import V21Loss


def _view(segmentation=0.0, presence=0.0, *, dtype=torch.float32):
    return {
        "segmentation": torch.full((2, 2, 3, 4), segmentation, dtype=dtype, requires_grad=True),
        "presence": torch.full((2, 2), presence, dtype=dtype, requires_grad=True),
    }


def _labels(*, positive=False):
    target = torch.zeros(2, 2, 3, 4)
    presence = torch.zeros(2, 2)
    if positive:
        target[0, 0, 1, 2] = 1
        target[1, 1, 2, 3] = 1
        presence[0, 0] = presence[1, 1] = 1
    return target, torch.ones(2, 3, 4, dtype=torch.bool), presence, torch.ones(2, 2, dtype=torch.bool)


def _assert_parts(parts):
    for value in parts.values():
        assert value.ndim == 0
        assert value.dtype == torch.float32
        assert torch.isfinite(value)
        assert not value.requires_grad
        assert value.grad_fn is None


@pytest.mark.parametrize("weight", [0.0, 0.05, 0.10, 0.20])
def test_identical_views_have_zero_consistency_and_preserve_v2_supervision(weight):
    labels = _labels(positive=True)
    original, second = _view(0.7, -0.3), _view(0.7, -0.3)
    options = dict(bce_weight=0.8, dice_weight=1.3, presence_weight=0.4, focal_gamma=2.0)
    total, parts = V21Loss(weight, **options)(original, second, *labels)
    expected, base_parts = V2Loss(**options)(original, *labels)
    torch.testing.assert_close(total, expected)
    for key, value in base_parts.items():
        torch.testing.assert_close(parts[key], value)
    for key in ("consistency_segmentation", "consistency_presence", "consistency", "weighted_consistency"):
        assert parts[key].item() == 0.0
    _assert_parts(parts)


@pytest.mark.parametrize("weight", [0.05, 0.10, 0.20])
def test_perturbation_adds_exact_probability_mse_to_average_supervision(weight):
    target, valid, presence, presence_valid = _labels(positive=True)
    valid.zero_()
    valid[0, 1, 2] = True
    presence_valid.zero_()
    presence_valid[1, 1] = True
    labels = target, valid, presence, presence_valid
    original, second = _view(), _view(math.log(3), math.log(3))
    total, parts = V21Loss(weight)(original, second, *labels)
    left, _ = V2Loss()(original, *labels)
    right, _ = V2Loss()(second, *labels)
    expected_mse = torch.tensor((0.75 - 0.5) ** 2)
    torch.testing.assert_close(parts["supervised_original"], left)
    torch.testing.assert_close(parts["supervised_photometric"], right)
    torch.testing.assert_close(parts["supervised"], (left + right) / 2)
    torch.testing.assert_close(parts["consistency_segmentation"], expected_mse)
    torch.testing.assert_close(parts["consistency_presence"], expected_mse)
    torch.testing.assert_close(total, (left + right) / 2 + weight * 2 * expected_mse)
    assert total > parts["supervised"]
    _assert_parts(parts)


def test_invalid_pixel_regions_and_presence_have_no_consistency_or_gradient():
    target, valid, presence, presence_valid = _labels(positive=True)
    valid[:, :, 2:] = False
    presence_valid[:, 1] = False
    original, second = _view(), _view()
    with torch.no_grad():
        second["segmentation"][:, :, :, 2:] = 9
        second["presence"][:, 1] = -9
    total, parts = V21Loss()(original, second, target, valid, presence, presence_valid)
    assert parts["consistency"].item() == 0.0
    expected, _ = V2Loss()(original, target, valid, presence, presence_valid)
    torch.testing.assert_close(total, expected)
    total.backward()
    for view in (original, second):
        assert torch.count_nonzero(view["segmentation"].grad[:, :, :, 2:]) == 0
        assert torch.count_nonzero(view["presence"].grad[:, 1]) == 0


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("pixel_valid,pair_valid", [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_finite_gradients_to_both_views_including_empty_labels_and_zero_validity(
    positive, pixel_valid, pair_valid, dtype,
):
    target, valid, presence, presence_valid = _labels(positive=positive)
    valid.fill_(pixel_valid)
    presence_valid.fill_(pair_valid)
    original, second = _view(-0.6, 0.4, dtype=dtype), _view(0.8, -0.7, dtype=dtype)
    total, parts = V21Loss()(original, second, target, valid, presence, presence_valid)
    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    _assert_parts(parts)
    total.backward()
    for view in (original, second):
        for key, active in (("segmentation", pixel_valid), ("presence", pair_valid)):
            grad = view[key].grad
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert (torch.count_nonzero(grad).item() > 0) == active
    if not pixel_valid:
        assert parts["consistency_segmentation"].item() == 0.0
    if not pair_valid:
        assert parts["consistency_presence"].item() == 0.0
    if not pixel_valid and not pair_valid:
        assert total.item() == 0.0
        assert all(value.item() == 0.0 for value in parts.values())


def test_consistency_itself_backpropagates_to_both_views():
    original, second = _view(-0.4, -0.8), _view(0.4, 0.8)
    total, parts = V21Loss(bce_weight=0, dice_weight=0, presence_weight=0)(
        original, second, *_labels(),
    )
    torch.testing.assert_close(total, parts["weighted_consistency"])
    total.backward()
    for key in ("segmentation", "presence"):
        assert torch.all(original[key].grad < 0)
        assert torch.all(second[key].grad > 0)
        torch.testing.assert_close(original[key].grad, -second[key].grad)


def test_positive_supervision_penalizes_agreed_all_zero_collapse_in_both_views():
    target = torch.ones(2, 2, 3, 4)
    presence = torch.ones(2, 2)
    _, valid, _, presence_valid = _labels()
    labels = target, valid, presence, presence_valid
    original, second = _view(-8, -8), _view(-8, -8)
    collapsed, parts = V21Loss()(original, second, *labels)
    preserved, _ = V21Loss()(_view(8, 8), _view(8, 8), *labels)
    assert parts["consistency"].item() == 0
    assert collapsed > preserved + 1
    collapsed.backward()
    for view in (original, second):
        assert torch.all(view["segmentation"].grad < 0)
        assert torch.all(view["presence"].grad < 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_extreme_logits_use_finite_float32_probability_math(dtype):
    original, second = _view(-1000, 1000, dtype=dtype), _view(1000, -1000, dtype=dtype)
    total, parts = V21Loss()(original, second, *_labels(positive=True))
    _assert_parts(parts)
    assert parts["consistency_segmentation"].item() == 1
    assert parts["consistency_presence"].item() == 1
    total.backward()
    for view in (original, second):
        for logits in view.values():
            assert torch.isfinite(logits.grad).all()


def test_expanded_pixel_mask_and_boolean_labels_preserve_v2_contract():
    target, valid, presence, presence_valid = _labels(positive=True)
    valid[:, 0] = False
    original, second = _view(-0.4, 0.2), _view(0.8, -0.5)
    expected, expected_parts = V21Loss()(original, second, target, valid, presence, presence_valid)
    actual, parts = V21Loss()(
        original, second, target.bool(), valid[:, None].expand_as(target), presence.bool(), presence_valid,
    )
    torch.testing.assert_close(actual, expected)
    for key in parts:
        torch.testing.assert_close(parts[key], expected_parts[key])


def test_view_order_is_symmetric():
    original, second = _view(-1, 0.3), _view(0.6, -0.2)
    forward, left = V21Loss()(original, second, *_labels(positive=True))
    reverse, right = V21Loss()(second, original, *_labels(positive=True))
    torch.testing.assert_close(forward, reverse)
    for key in ("supervised", "consistency_segmentation", "consistency_presence", "consistency"):
        torch.testing.assert_close(left[key], right[key])


@pytest.mark.parametrize("weight", [-0.01, 0.20001, math.inf, -math.inf, math.nan])
def test_invalid_consistency_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="consistency_weight"):
        V21Loss(weight)


@pytest.mark.parametrize("view_index,key,shape", [
    (0, "segmentation", (2, 1, 3, 4)),
    (1, "segmentation", (2, 2, 3, 1)),
    (1, "segmentation", (2, 2, 3)),
    (0, "presence", (2, 1)),
    (1, "presence", (1, 2)),
    (1, "presence", (2, 2, 1)),
])
def test_logits_shape_errors_are_rejected_before_broadcasting(view_index, key, shape):
    views = [_view(), _view()]
    views[view_index][key] = torch.zeros(shape)
    with pytest.raises(ValueError, match=f"{key} logits"):
        V21Loss()(*views, *_labels())


@pytest.mark.parametrize("label_index,shape,message", [
    (0, (2, 3, 4), "target"),
    (0, (2, 1, 3, 4), "target"),
    (1, (2, 1, 3, 4), "valid_mask"),
    (1, (1, 3, 4), "valid_mask"),
    (1, (2, 3, 1), "valid_mask"),
    (2, (2, 1), "presence"),
    (3, (2,), "presence_valid"),
    (3, (2, 1), "presence_valid"),
])
def test_label_and_mask_shape_errors_are_rejected(label_index, shape, message):
    labels = list(_labels())
    labels[label_index] = torch.zeros(shape)
    with pytest.raises(ValueError, match=message):
        V21Loss()(_view(), _view(), *labels)


def test_per_class_pixel_validity_is_rejected_instead_of_silently_collapsing_masks():
    target, valid, presence, presence_valid = _labels()
    expanded = valid[:, None].expand_as(target).clone()
    expanded[0, 1, 0, 0] = False
    with pytest.raises(ValueError, match="identical channels"):
        V21Loss()(_view(), _view(), target, expanded, presence, presence_valid)
