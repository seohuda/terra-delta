import numpy as np
import pytest
from shapely import Polygon

from terradelta.data.dataset_v2 import _tensor
from terradelta.inference.v2 import prepare_v2_pair
from terradelta.inference.v2_calibration import PIXEL, PRESENCE, select_plateau, statistics
from terradelta.metrics.evaluation import _shape_f1, evaluate_predictions


def test_cached_statistics_match_exported_polygon_evaluator():
    square = [[10, 10], [30, 10], [30, 30], [10, 30]]
    shifted = [[12, 10], [32, 10], [32, 30], [12, 30]]
    gt = [square, square, None, None]
    pred = [shifted, None, square, None]
    truth = [{"id": str(i), "new_building": [g] if g else "", "tree_removal": ""} for i, g in enumerate(gt)]
    rows = [{"id": str(i), "new_building": [p] if p else "", "tree_removal": ""} for i, p in enumerate(pred)]
    shapes = np.array([_shape_f1(Polygon(shifted), Polygon(square)), 0, 0, 0])
    fast = statistics(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
        shapes,
        np.array([False, False, True, True]),
    )
    exact = evaluate_predictions(rows, truth)["classes"]["new_building"]
    for name, value in exact.items():
        assert fast[name] == pytest.approx(value) if isinstance(value, float) else fast[name] == value
    assert fast["no_change_fp_count"] == 1


def test_calibration_and_submission_rgb_normalization_identical():
    import torch

    rng = np.random.default_rng(9)
    pre, post = rng.integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    torch.testing.assert_close(
        prepare_v2_pair(pre, post), torch.cat([_tensor(pre), _tensor(post)]), rtol=0, atol=0
    )


def test_plateau_prefers_stability_over_isolated_peak():
    rows = []
    for presence in (*PRESENCE, None):
        for pixel in PIXEL:
            utility = 0.59 if presence == 0.25 and 0.45 <= pixel <= 0.55 else 0.2
            if presence in (0.20, 0.30) and 0.45 <= pixel <= 0.55:
                utility = 0.59
            if presence == 0.45 and pixel == 0.65:
                utility = 0.60
            metrics = {"score": utility, "no_change_fp_count": 0}
            rows.append(
                {"presence": presence, "pixel": pixel, "legacy": dict(metrics), "stress": dict(metrics)}
            )
    selected = select_plateau(rows)
    assert selected["presence"] == 0.25
    assert selected["pixel"] == 0.50


def test_grid_matches_real_output_cells():
    import torch
    from terradelta.inference.v2 import output_row
    from terradelta.inference.v2_calibration import evaluate_grid
    from terradelta.postprocess.polygons import mask_to_polygons_reference

    target = torch.zeros(2, 256, 256)
    target[0, 20:40, 20:40] = 1
    samples = [
        {"id": "positive", "target": target, "valid_mask": torch.ones(256, 256, dtype=torch.bool)},
        {
            "id": "negative",
            "target": torch.zeros_like(target),
            "valid_mask": torch.ones(256, 256, dtype=torch.bool),
        },
    ]
    pixels = np.stack([target.numpy() * 0.8, np.zeros_like(target.numpy())])
    presence = np.ones((2, 2), np.float32)
    trials = evaluate_grid(samples, pixels, presence)
    truth = [
        {
            "id": s["id"],
            **{
                n: mask_to_polygons_reference(s["target"][c], min_area=0, min_pos_area=0, simplify_px=0)
                for c, n in enumerate(("new_building", "tree_removal"))
            },
        }
        for s in samples
    ]
    for i, trial in enumerate(trials["new_building"]):
        config = {"postprocess": {"presence_threshold": trial["presence"], "pixel_threshold": trial["pixel"]}}
        rows = [output_row(s["id"], p, q, config) for s, p, q in zip(samples, pixels, presence)]
        exact = evaluate_predictions(rows, truth)
        for name in ("new_building", "tree_removal"):
            assert trials[name][i]["metrics"]["score"] == exact["classes"][name]["score"]
