"""Local implementation of the published polygon scoring rules."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Any

from shapely import GeometryCollection, Polygon, union_all
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

CHANGE_CLASSES = ("new_building", "tree_removal")
MIN_PREDICTION_AREA = 20.0


def _empty_cell(value: Any) -> bool:
    return (
        value is None
        or isinstance(value, str) and value.strip().lower() in ("", "nan")
        or isinstance(value, Real) and math.isnan(value)
    )


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _polygon_union(value: Any, context: str) -> BaseGeometry:
    if _empty_cell(value):
        return GeometryCollection()
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_constant=_invalid_json_constant)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{context}: invalid polygon JSON: {exc}") from exc
        if value is None:
            return GeometryCollection()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context}: expected a list of polygon exteriors")

    polygons = []
    for index, exterior in enumerate(value):
        label = f"{context}, polygon {index}"
        if not isinstance(exterior, (list, tuple)) or len(exterior) < 3:
            raise ValueError(f"{label}: exterior must contain at least three points")
        coordinates = []
        for point in exterior:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
                raise ValueError(f"{label}: each point must be an [x, y] pair")
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in point):
                raise ValueError(f"{label}: coordinates must be finite real numbers")
            coordinates.append(tuple(point))
        try:
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                raise ValueError(f"{label}: invalid polygon: {explain_validity(polygon)}")
            if polygon.is_empty or not math.isfinite(polygon.area) or polygon.area <= 0:
                raise ValueError(f"{label}: polygon must have finite positive area")
        except GEOSException as exc:
            raise ValueError(f"{label}: invalid polygon: {exc}") from exc
        polygons.append(polygon)
    try:
        geometry = union_all(polygons)
        if not geometry.is_valid or not math.isfinite(geometry.area):
            raise ValueError(f"{context}: polygon union must be valid with finite area")
    except GEOSException as exc:
        raise ValueError(f"{context}: polygon union failed: {exc}") from exc
    return geometry


def _index_rows(rows: Iterable[Mapping[str, Any]], source: str) -> dict[Any, dict[str, BaseGeometry]]:
    indexed = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source} row {position}: expected a mapping")
        missing = {"id", *CHANGE_CLASSES} - row.keys()
        if missing:
            raise ValueError(f"{source} row {position}: missing fields {sorted(missing)}")
        row_id = row["id"]
        if _empty_cell(row_id) or isinstance(row_id, Real) and not math.isfinite(row_id):
            raise ValueError(f"{source} row {position}: id must be nonempty and finite")
        try:
            hash(row_id)
        except TypeError as exc:
            raise ValueError(f"{source} row {position}: id must be hashable") from exc
        if row_id in indexed:
            raise ValueError(f"{source}: duplicate id {row_id!r}")
        indexed[row_id] = {
            name: _polygon_union(row[name], f"{source} id {row_id!r}, {name}")
            for name in CHANGE_CLASSES
        }
    return indexed


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _shape_f1(prediction: BaseGeometry, ground_truth: BaseGeometry) -> float:
    if prediction.is_empty:
        return 0.0
    precision = _ratio(prediction.intersection(ground_truth.buffer(1, join_style="mitre")).area, prediction.area)
    recall = _ratio(ground_truth.intersection(prediction.buffer(1, join_style="mitre")).area, ground_truth.area)
    return _ratio(2 * precision * recall, precision + recall)


def evaluate_predictions(
    pred_rows: Iterable[Mapping[str, Any]], gt_rows: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score exported exteriors for both change classes, matching rows by ``id``.

    Each row requires ``id``, ``new_building``, and ``tree_removal``. Cells are
    lists of exterior rings or JSON encoding those lists; unclosed and closed
    rings are accepted. Empty CSV cells (None, NaN, or blank strings) are empty.
    Invalid polygons and duplicate/mismatched IDs raise ValueError.

    Returns ``score``, ``num_samples``, and ``classes``. Each class contains
    ``presence_macro_f1``, ``shape_score``, ``score``, ``gt_positive_count``,
    and ``presence_confusion`` (tp, fp, fn, tn). Zero denominators score zero.
    """
    predictions = _index_rows(pred_rows, "prediction")
    ground_truth = _index_rows(gt_rows, "ground truth")
    missing = ground_truth.keys() - predictions.keys()
    extra = predictions.keys() - ground_truth.keys()
    if missing or extra:
        raise ValueError(
            f"prediction IDs differ from ground truth: missing={sorted(map(repr, missing))}, "
            f"extra={sorted(map(repr, extra))}"
        )

    classes = {}
    for name in CHANGE_CLASSES:
        tp = fp = fn = tn = positive_count = 0
        shape_scores = []
        for row_id, truth in ground_truth.items():
            gt = truth[name]
            pred = predictions[row_id][name]
            if pred.area < MIN_PREDICTION_AREA:
                pred = GeometryCollection()
            gt_positive, pred_positive = not gt.is_empty, not pred.is_empty
            tp += int(gt_positive and pred_positive)
            fp += int(not gt_positive and pred_positive)
            fn += int(gt_positive and not pred_positive)
            tn += int(not gt_positive and not pred_positive)
            if gt_positive:
                positive_count += 1
                shape_scores.append(_shape_f1(pred, gt))
        positive_f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        negative_f1 = _ratio(2 * tn, 2 * tn + fp + fn)
        macro_f1 = (positive_f1 + negative_f1) / 2
        shape_score = _ratio(math.fsum(shape_scores), positive_count)
        classes[name] = {
            "presence_macro_f1": macro_f1,
            "shape_score": shape_score,
            "score": (macro_f1 + shape_score) / 2,
            "gt_positive_count": positive_count,
            "presence_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
    return {
        "score": sum(result["score"] for result in classes.values()) / len(CHANGE_CLASSES),
        "num_samples": len(ground_truth),
        "classes": classes,
    }
