"""Diagnostic raster overlap and boundary metrics; these are not leaderboard scores."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _binary_pair(prediction: ArrayLike, ground_truth: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    pred, gt = np.asarray(prediction), np.asarray(ground_truth)
    if pred.shape != gt.shape or pred.ndim == 0:
        raise ValueError("masks must have equal, nonscalar shapes")
    for mask in (pred, gt):
        if mask.dtype.kind not in "biuf" or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("binary masks must contain only 0/1 or boolean values")
    return pred.astype(bool), gt.astype(bool)


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _overlap(tp: int, fp: int, fn: int, tn: int) -> dict[str, int | float]:
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": _divide(tp, tp + fp + fn),
        "dice": f1,
        "f1": f1,
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
        "accuracy": _divide(tp + tn, tp + fp + fn + tn),
    }


def binary_segmentation_metrics(prediction: ArrayLike, ground_truth: ArrayLike) -> dict[str, int | float]:
    """Pool all pixels of equal-shaped 0/1 masks; zero denominators return zero.

    Batched masks are supported for overlap. For independent overlapping change
    channels, call separately on each channel rather than converting to argmax.
    """
    pred, gt = _binary_pair(prediction, ground_truth)
    return _overlap(
        int(np.count_nonzero(pred & gt)), int(np.count_nonzero(pred & ~gt)),
        int(np.count_nonzero(~pred & gt)), int(np.count_nonzero(~pred & ~gt)),
    )


def _label_pair(
    prediction: ArrayLike, ground_truth: ArrayLike, num_classes: int, ignore_index: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral)):
        raise ValueError("ignore_index must be an integer or None")
    pred, gt = np.asarray(prediction), np.asarray(ground_truth)
    if pred.shape != gt.shape or pred.ndim == 0:
        raise ValueError("label maps must have equal, nonscalar shapes")
    if pred.dtype.kind not in "iu" or gt.dtype.kind not in "iu":
        raise ValueError("multiclass label maps must contain integers")
    valid = np.ones(gt.shape, dtype=bool) if ignore_index is None else gt != ignore_index
    for labels in (pred[valid], gt[valid]):
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError("labels on evaluated pixels must be in [0, num_classes)")
    return pred, gt, valid


def multiclass_segmentation_metrics(
    prediction: ArrayLike, ground_truth: ArrayLike, num_classes: int, *, ignore_index: int | None = None
) -> dict[str, Any]:
    """Pool exclusive integer labels; confusion rows are GT, columns prediction.

    Macro means include every declared class, including background and absent
    classes. GT ignore_index pixels are excluded from all counts.
    """
    pred, gt, valid = _label_pair(prediction, ground_truth, num_classes, ignore_index)
    encoded = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    confusion = np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    total = int(confusion.sum())
    per_class = {}
    for label in range(num_classes):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum()) - tp
        fn = int(confusion[label, :].sum()) - tp
        per_class[label] = _overlap(tp, fp, fn, total - tp - fp - fn)
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "macro_iou": sum(m["iou"] for m in per_class.values()) / num_classes,
        "macro_dice": sum(m["dice"] for m in per_class.values()) / num_classes,
        "macro_f1": sum(m["f1"] for m in per_class.values()) / num_classes,
        "accuracy": _divide(int(np.trace(confusion)), total),
        "num_pixels": total,
    }


def boundary_metrics(
    prediction: ArrayLike, ground_truth: ArrayLike, *, tolerance: float = 1.0
) -> dict[str, int | float]:
    """Match inner boundary pixels of two 2-D binary masks within Euclidean tolerance.

    Boundary = mask minus 3x3 erosion with outside-image background. Match counts
    are directional and many-to-many. Empty boundary denominators return zero.
    Requires SciPy; imports it only when this diagnostic is called.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")
    pred, gt = _binary_pair(prediction, ground_truth)
    if pred.ndim != 2 or 0 in pred.shape:
        raise ValueError("boundary metrics require nonempty 2-D masks")
    from scipy.ndimage import binary_erosion, distance_transform_edt

    structure = np.ones((3, 3), dtype=bool)
    pred_boundary = pred & ~binary_erosion(pred, structure=structure, border_value=0)
    gt_boundary = gt & ~binary_erosion(gt, structure=structure, border_value=0)
    pred_count, gt_count = int(pred_boundary.sum()), int(gt_boundary.sum())
    matched_pred = matched_gt = 0
    if pred_count and gt_count:
        matched_pred = int(np.count_nonzero(distance_transform_edt(~gt_boundary)[pred_boundary] <= tolerance))
        matched_gt = int(np.count_nonzero(distance_transform_edt(~pred_boundary)[gt_boundary] <= tolerance))
    precision, recall = _divide(matched_pred, pred_count), _divide(matched_gt, gt_count)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
        "predicted_boundary_pixels": pred_count,
        "gt_boundary_pixels": gt_count,
        "matched_predicted_boundary_pixels": matched_pred,
        "matched_gt_boundary_pixels": matched_gt,
    }


def multiclass_boundary_metrics(
    prediction: ArrayLike, ground_truth: ArrayLike, num_classes: int, *, tolerance: float = 1.0
) -> dict[str, Any]:
    """One-vs-rest boundary metrics for exclusive 2-D integer labels.

    Includes background and absent classes in macro_f1. No ignored-label support:
    removing ignored pixels could introduce artificial boundaries.
    """
    pred, gt, _ = _label_pair(prediction, ground_truth, num_classes, None)
    per_class = {
        label: boundary_metrics(pred == label, gt == label, tolerance=tolerance)
        for label in range(num_classes)
    }
    return {"per_class": per_class, "macro_f1": sum(m["f1"] for m in per_class.values()) / num_classes}
