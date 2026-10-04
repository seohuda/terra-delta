"""Hand-computed scoring cases and strict input-validation regressions."""

import copy
import json
import math

import numpy as np
import pytest

from terradelta.metrics import (
    CHANGE_CLASSES,
    MIN_PREDICTION_AREA,
    binary_segmentation_metrics,
    boundary_metrics,
    evaluate_predictions,
    multiclass_boundary_metrics,
    multiclass_segmentation_metrics,
)


def rectangle(x=0, y=0, width=10, height=10, *, closed=False):
    ring = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    return ring + [ring[0]] if closed else ring


def row(row_id, new_building=None, tree_removal=None):
    return {"id": row_id, "new_building": new_building, "tree_removal": tree_removal}


def both(row_id, polygons):
    return row(row_id, polygons, polygons)


def test_public_constants():
    assert CHANGE_CLASSES == ("new_building", "tree_removal")
    assert MIN_PREDICTION_AREA == 20


def test_competition_compatibility_imports():
    from terradelta.metrics.competition import (
        CHANGE_CLASSES as classes,
        MIN_PREDICTION_AREA as minimum_area,
        evaluate_predictions as evaluate,
    )

    assert classes == CHANGE_CLASSES
    assert minimum_area == MIN_PREDICTION_AREA
    assert evaluate is evaluate_predictions


def test_reference_list_and_serialized_postprocess_exports_score_identically():
    from terradelta.postprocess.polygons import (
        mask_to_polygons_reference,
        predictions_to_polygons,
        serialize_polygons,
    )

    # Exported exteriors fill this hole. Evaluation must use that exported
    # geometry (100 pixels), not the original mask's foreground area (64).
    mask = np.ones((10, 10), dtype=bool)
    mask[2:8, 2:8] = False
    probabilities = np.zeros((3, 10, 10), dtype=float)
    probabilities[0] = ~mask
    probabilities[1] = mask
    exports = predictions_to_polygons(probabilities)
    gt = [row("positive", [rectangle()]), row("negative")]
    reference = mask_to_polygons_reference(mask)
    assert json.loads(reference) == exports["new_building"]
    predictions = [
        [row("positive", reference), row("negative")],
        [{"id": "positive", **exports}, row("negative")],
        [{"id": "positive", **{name: serialize_polygons(value) for name, value in exports.items()}}, row("negative")],
    ]
    for pred in predictions:
        result = evaluate_predictions(pred, gt)
        assert result["classes"]["new_building"]["shape_score"] == 1
        assert result["score"] == 0.625


def test_perfect_mixed_dataset_json_lists_and_overlapping_classes():
    gt = [both("positive", [rectangle()]), both("negative", [])]
    pred = [both("negative", ""), both("positive", json.dumps([rectangle(closed=True)]))]
    original_gt, original_pred = copy.deepcopy(gt), copy.deepcopy(pred)
    result = evaluate_predictions(iter(pred), (r for r in gt))
    assert result["score"] == 1
    assert result["num_samples"] == 2
    for metrics in result["classes"].values():
        assert metrics == {
            "presence_macro_f1": 1, "shape_score": 1, "score": 1,
            "gt_positive_count": 1, "presence_confusion": {"tp": 1, "fp": 0, "fn": 0, "tn": 1},
        }
    assert (gt, pred) == (original_gt, original_pred)
    json.dumps(result, allow_nan=False)


def test_class_and_layer_weights_are_equal():
    gt = [row("p", [rectangle()]), row("n")]
    result = evaluate_predictions(gt, gt)
    assert result["classes"]["new_building"]["score"] == 1
    assert result["classes"]["tree_removal"]["presence_macro_f1"] == 0.5
    assert result["classes"]["tree_removal"]["shape_score"] == 0
    assert result["score"] == 0.625


def test_false_positive_and_missed_positive_penalize_presence_and_shape():
    gt = [both("tp", [rectangle()]), both("fn", [rectangle()]), both("fp", []), both("tn", [])]
    pred = [both("tp", [rectangle()]), both("fn", []), both("fp", [rectangle()]), both("tn", [])]
    result = evaluate_predictions(pred, gt)
    for metrics in result["classes"].values():
        assert metrics["presence_confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
        assert metrics["presence_macro_f1"] == 0.5
        assert metrics["shape_score"] == 0.5  # denominator includes the absent prediction
    assert result["score"] == 0.5


def test_macro_f1_is_unweighted_not_accuracy_or_positive_only():
    gt = [both("tp", [rectangle()]), both("fn", [rectangle()])] + [both(i, []) for i in range(3)]
    pred = [both("tp", [rectangle()]), both("fn", [])] + [both(i, []) for i in range(3)]
    result = evaluate_predictions(pred, gt)
    assert result["classes"]["new_building"]["presence_macro_f1"] == pytest.approx(16 / 21)
    assert result["score"] == pytest.approx(53 / 84)


@pytest.mark.parametrize("area,expected_presence,expected_shape", [(19.999, False, 0), (20, True, 1)])
def test_area_threshold_is_inclusive_and_applies_to_both_layers(area, expected_presence, expected_shape):
    polygons = [rectangle(width=area / 5, height=5)]
    result = evaluate_predictions([both("p", polygons)], [both("p", polygons)])
    metrics = result["classes"]["new_building"]
    assert bool(metrics["presence_confusion"]["tp"]) is expected_presence
    assert metrics["shape_score"] == expected_shape
    assert metrics["gt_positive_count"] == 1  # GT does not use the 20-pixel threshold


def test_union_area_not_sum_of_overlapping_polygon_areas():
    small = rectangle(width=4, height=4)
    gt = [both("p", [small])]
    result = evaluate_predictions([both("p", [small, small])], gt)
    assert result["score"] == 0  # area 16, not 32
    # Two distinct subthreshold components jointly exceed the threshold.
    separate = [rectangle(width=3, height=4), rectangle(x=20, width=3, height=4)]
    result = evaluate_predictions([both("p", separate)], [both("p", separate)])
    assert result["classes"]["new_building"]["shape_score"] == 1
    assert result["classes"]["new_building"]["presence_confusion"]["tp"] == 1


def test_overlapping_ground_truth_and_prediction_exteriors_are_unioned():
    gt = [both("p", [rectangle(), rectangle(x=5)])]
    pred = [both("p", [rectangle(width=15)])]
    assert evaluate_predictions(pred, gt)["classes"]["new_building"]["shape_score"] == 1


@pytest.mark.parametrize("dx,dy,shape", [(1, 0, 1), (1, 1, 1), (2, 0, 0.9), (2, 2, 0.81), (30, 0, 0)])
def test_mitre_dilation_tolerates_axis_and_corner_offsets(dx, dy, shape):
    gt = [both("p", [rectangle()])]
    pred = [both("p", [rectangle(x=dx, y=dy)])]
    result = evaluate_predictions(pred, gt)
    assert result["classes"]["new_building"]["shape_score"] == pytest.approx(shape)


def test_shape_uses_two_directional_area_ratios_and_harmonic_mean():
    # GT area 100, prediction 200. GT dilation meets prediction over area 110:
    # precision .55, recall 1, harmonic mean 22/31 (not IoU .5).
    result = evaluate_predictions([both("p", [rectangle(width=20)])], [both("p", [rectangle()])])
    assert result["classes"]["new_building"]["shape_score"] == pytest.approx(22 / 31)


def test_shape_is_mean_over_pairs_not_weighted_by_gt_area():
    gt = [both("small", [rectangle(width=5, height=5)]), both("large", [rectangle(width=100, height=100)])]
    pred = [both("small", [rectangle(width=5, height=5)]), both("large", [])]
    assert evaluate_predictions(pred, gt)["classes"]["new_building"]["shape_score"] == 0.5


@pytest.mark.parametrize("empty", [None, "", "  ", float("nan"), np.float32("nan"), "nan", "NaN", [], "[]", "null"])
def test_csv_empty_cells(empty):
    result = evaluate_predictions([both("n", empty)], [both("n", [])])
    assert result["score"] == 0.25
    assert result["classes"]["new_building"]["presence_confusion"]["tn"] == 1


def test_absent_label_and_no_gt_positive_conventions():
    positive = [both("p", [rectangle()])]
    result = evaluate_predictions(positive, positive)
    assert result["classes"]["new_building"]["presence_macro_f1"] == 0.5
    assert result["score"] == 0.75
    result = evaluate_predictions([both("n", [rectangle()])], [both("n", [])])
    assert result["score"] == 0
    assert result["classes"]["new_building"]["shape_score"] == 0
    assert evaluate_predictions([], [])["score"] == 0


@pytest.mark.parametrize("source", ["prediction", "ground truth"])
def test_duplicate_ids_are_rejected(source):
    rows = [both("p", []), both("p", [])]
    pred, gt = (rows, [rows[0]]) if source == "prediction" else ([rows[0]], rows)
    with pytest.raises(ValueError, match=f"{source}: duplicate id"):
        evaluate_predictions(pred, gt)


@pytest.mark.parametrize("pred_ids,gt_ids", [([], ["a"]), (["b"], []), (["b"], ["a"])])
def test_missing_and_extra_ids_are_rejected(pred_ids, gt_ids):
    with pytest.raises(ValueError, match="missing=.*extra="):
        evaluate_predictions([both(i, []) for i in pred_ids], [both(i, []) for i in gt_ids])


@pytest.mark.parametrize("bad_id", [None, "", " ", float("nan"), float("inf"), []])
def test_invalid_ids(bad_id):
    with pytest.raises(ValueError, match="id must"):
        evaluate_predictions([both(bad_id, [])], [])


@pytest.mark.parametrize("bad_row", [{"id": "p"}, {"new_building": [], "tree_removal": []}, []])
def test_invalid_row_schema(bad_row):
    with pytest.raises(ValueError, match="missing fields|expected a mapping"):
        evaluate_predictions([bad_row], [])


@pytest.mark.parametrize("bad", [
    "not JSON", "{}", "42", "Infinity", 0, float("inf"),
    [[0, 0], [1, 0], [0, 1]],  # ring supplied without the polygon-list level
    [[]], [[[0, 0], [1, 1]]],
    [[[0, 0], [10, 10], [0, 10], [10, 0]]],  # bow tie, never repaired
    [[[0, 0], [1, 1], [2, 2]]],  # collinear
    [[[0, 0], [1, 0], [math.nan, 1]]],
    '[[[0, 0], [1, 0], [Infinity, 1]]]',
    [[[0, 0], [1, 0], ["1", 1]]],
    [[[0, 0], [1, 0], [True, 1]]],
    [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
    [[[[0, 0], [10, 0], [10, 10]], [[1, 1], [2, 1], [2, 2]]]],  # holes not supported
])
@pytest.mark.parametrize("source", ["prediction", "ground truth"])
def test_invalid_polygons_fail_with_row_and_class_context(bad, source):
    invalid, valid = [row("bad-id", bad)], [row("bad-id")]
    pred, gt = (invalid, valid) if source == "prediction" else (valid, invalid)
    with pytest.raises(ValueError, match=f"{source} id 'bad-id', new_building"):
        evaluate_predictions(pred, gt)


def test_binary_overlap_hand_computed():
    result = binary_segmentation_metrics([1, 1, 0, 0, 0], [1, 0, 1, 1, 0])
    assert result == {
        "tp": 1, "fp": 1, "fn": 2, "tn": 1,
        "iou": 0.25, "dice": 0.4, "f1": 0.4,
        "precision": 0.5, "recall": 1 / 3, "accuracy": 0.4,
    }


def test_binary_batched_overlap_and_overlapping_channels():
    mask = np.ones((2, 3, 4), dtype=bool)
    assert binary_segmentation_metrics(mask, mask)["tp"] == 24
    for channel in mask:
        assert binary_segmentation_metrics(channel, channel)["dice"] == 1


def test_binary_empty_denominators():
    result = binary_segmentation_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
    assert (result["iou"], result["dice"], result["precision"], result["recall"]) == (0, 0, 0, 0)
    assert result["accuracy"] == 1
    assert binary_segmentation_metrics([], [])["accuracy"] == 0


@pytest.mark.parametrize("pred,gt", [([1], [1, 0]), ([0.2], [0]), ([np.nan], [0]), ([2], [0]), (["1"], [0]), (1, 1)])
def test_binary_validation(pred, gt):
    with pytest.raises(ValueError):
        binary_segmentation_metrics(pred, gt)


def test_multiclass_confusion_orientation_and_macros():
    # GT rows: [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    result = multiclass_segmentation_metrics([0, 1, 1, 2, 2, 0], [0, 0, 1, 1, 2, 2], 3)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert result["accuracy"] == 0.5
    assert result["num_pixels"] == 6
    assert result["macro_iou"] == pytest.approx(1 / 3)
    assert result["macro_dice"] == result["macro_f1"] == 0.5
    assert result["per_class"][0]["precision"] == result["per_class"][0]["recall"] == 0.5


def test_multiclass_ignored_pixels_and_declared_absent_classes():
    result = multiclass_segmentation_metrics([0, 1, 999], [0, 1, -1], 3, ignore_index=-1)
    assert result["num_pixels"] == 2
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert result["macro_iou"] == result["macro_dice"] == pytest.approx(2 / 3)
    assert result["per_class"][2]["dice"] == 0
    assert multiclass_segmentation_metrics([99], [-1], 2, ignore_index=-1)["accuracy"] == 0
    empty = np.zeros((0, 2), dtype=int)
    assert multiclass_segmentation_metrics(empty, empty, 2)["macro_f1"] == 0


@pytest.mark.parametrize("pred,gt,classes,kwargs", [
    ([0], [0, 1], 2, {}), ([0.0], [0], 2, {}), ([True], [0], 2, {}),
    ([2], [0], 2, {}), ([0], [-1], 2, {}), ([0], [0], 0, {}),
    ([0], [0], 1.5, {}), ([0], [0], True, {}), ([0], [0], 2, {"ignore_index": 0.5}),
])
def test_multiclass_validation(pred, gt, classes, kwargs):
    with pytest.raises(ValueError):
        multiclass_segmentation_metrics(pred, gt, classes, **kwargs)


def test_boundary_perfect_full_mask_counts_image_edges():
    mask = np.ones((5, 5), dtype=bool)
    result = boundary_metrics(mask, mask, tolerance=0)
    assert result["predicted_boundary_pixels"] == result["gt_boundary_pixels"] == 16
    assert result["precision"] == result["recall"] == result["f1"] == 1


def test_boundary_holes_count_as_boundaries():
    mask = np.ones((7, 7), dtype=bool)
    mask[3, 3] = False
    assert boundary_metrics(mask, mask, tolerance=0)["gt_boundary_pixels"] == 32  # outer 24 + inner 8


def test_boundary_directional_matches_and_harmonic_mean():
    pred, gt = np.zeros((5, 5)), np.zeros((5, 5))
    pred[1, 1] = pred[3, 3] = gt[1, 1] = 1
    result = boundary_metrics(pred, gt, tolerance=0)
    assert result["precision"] == 0.5
    assert result["recall"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_boundary_tolerance_is_euclidean_not_square_dilation():
    pred, gt = np.zeros((5, 5)), np.zeros((5, 5))
    gt[1, 1], pred[2, 2] = 1, 1
    assert boundary_metrics(pred, gt, tolerance=1)["f1"] == 0
    assert boundary_metrics(pred, gt, tolerance=math.sqrt(2))["f1"] == 1
    pred[2, 2], pred[1, 2] = 0, 1
    assert boundary_metrics(pred, gt, tolerance=0)["f1"] == 0
    assert boundary_metrics(pred, gt, tolerance=1)["f1"] == 1


@pytest.mark.parametrize("pred_present,gt_present", [(False, False), (True, False), (False, True)])
def test_boundary_empty_denominators(pred_present, gt_present):
    pred, gt = np.zeros((3, 3)), np.zeros((3, 3))
    pred[1, 1], gt[1, 1] = pred_present, gt_present
    result = boundary_metrics(pred, gt)
    assert result["precision"] == result["recall"] == result["f1"] == 0


@pytest.mark.parametrize("tolerance", [-1, math.inf, math.nan, True, "1"])
def test_boundary_tolerance_validation(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        boundary_metrics(np.zeros((3, 3)), np.zeros((3, 3)), tolerance=tolerance)


@pytest.mark.parametrize("shape", [(4,), (1, 3, 3), (0, 3)])
def test_boundary_requires_nonempty_single_image(shape):
    with pytest.raises(ValueError, match="2-D"):
        boundary_metrics(np.zeros(shape), np.zeros(shape))


def test_multiclass_boundaries_include_background_and_absent_classes():
    labels = np.zeros((5, 5), dtype=int)
    labels[1:4, 1:4] = 1
    result = multiclass_boundary_metrics(labels, labels, 3, tolerance=0)
    assert result["per_class"][0]["f1"] == result["per_class"][1]["f1"] == 1
    assert result["per_class"][2]["f1"] == 0
    assert result["macro_f1"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="labels"):
        multiclass_boundary_metrics(labels, labels - 1, 3)
