"""Search saved probabilities using exported polygons and an explicit objective.

Grid format: ``{class_name: {parameter: [values, ...]}}``. Supported parameters
are threshold, min_area, min_pos_area, simplify_px. Cartesian explores the full
grid. Staged optimizes both classes jointly for one parameter at a time, retaining
the incumbent before each stage. Staged search is not a global-optimum claim.
"""

import csv
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from numbers import Real
from pathlib import Path

import numpy as np
import yaml

from terradelta.training.validation import CLASSES, check_probabilities, ground_truth_row, prediction_row, sample_id


PARAMETERS = ("threshold", "min_area", "min_pos_area", "simplify_px")
DEFAULTS = dict(threshold=0.5, min_area=30, min_pos_area=20, simplify_px=0.5)
DEFAULT_GRID = {
    name: dict(threshold=[0.4, 0.5, 0.6], min_area=[10, 30, 50],
               min_pos_area=[0, 20, 50], simplify_px=[0, 0.5, 1])
    for name in CLASSES
}


def load_probability_maps(pred_dir):
    files = sorted(Path(pred_dir).glob("*.npy"))
    if not files:
        raise ValueError(f"No .npy probability maps found in {pred_dir}")
    return {sample_id(path.stem): check_probabilities(
        np.load(path, allow_pickle=False, mmap_mode="r"), context=str(path),
    ) for path in files}


def load_ground_truth(path, *, kind="auto"):
    """Load polygon CSV or a labeled dataset manifest, preserving CSV string IDs.

    Manifest paths follow the data loader's contract. No split is performed: the
    manifest supplied to search must already represent the validation set.
    """
    path = Path(path)
    if kind not in {"auto", "polygons", "manifest"}:
        raise ValueError("ground-truth kind must be auto, polygons, or manifest")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError("Ground truth is empty")
    if kind == "auto":
        kind = "manifest" if {"pre", "post"} <= fields or {"pre_path", "post_path"} <= fields else "polygons"
    if kind == "polygons":
        if not {"id", *CLASSES} <= fields:
            raise ValueError("Polygon GT CSV requires id,new_building,tree_removal columns")
        return [{"id": sample_id(row["id"]), **{name: row[name] or "" for name in CLASSES}} for row in rows]
    from terradelta.data.dataset import ChangeDataset

    dataset = ChangeDataset(path, transform=None, require_masks=True)
    return [ground_truth_row(dataset[index]) for index in range(len(dataset))]


def objective_value(scores, objective):
    """Select a scalar evaluator key, optionally dotted (e.g. macro.score)."""
    value = scores
    for key in objective.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Objective {objective!r} is absent from evaluator scores; available: {list(scores)}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"Objective {objective!r} must be a finite numeric evaluator score")
    return float(value)


def _prepare(config, grid):
    from terradelta.postprocess import class_config

    config = deepcopy(dict(config or {}))
    postprocess = config.setdefault("postprocess", {})
    # Resolve inherited/direct/alias settings once so no older spelling silently
    # overrides a searched canonical key. Retain morphology and polygon options.
    classes = {name: class_config(config, name) for name in CLASSES}
    aliases = ("min_component_area", "min_positive_total_area", "min_total_area", "simplification_tolerance")
    for name in CLASSES:
        settings = classes[name]
        for alias in aliases:
            settings.pop(alias, None)
        for parameter, default in DEFAULTS.items():
            settings.setdefault(parameter, default)
        postprocess.pop(name, None)
    for alias in aliases:
        postprocess.pop(alias, None)
    postprocess["classes"] = classes
    postprocess["mode"] = "threshold"
    dimensions = []
    for name, values in grid.items():
        if name not in CLASSES or not isinstance(values, Mapping):
            raise ValueError(f"Invalid class in search grid: {name!r}")
        for parameter, candidates in values.items():
            if parameter not in PARAMETERS:
                raise ValueError(f"Unsupported search parameter: {parameter}")
            if isinstance(candidates, (str, bytes)):
                raise ValueError(f"Grid values for {name}.{parameter} must be a nonempty sequence")
            candidates = list(candidates)
            if not candidates:
                raise ValueError(f"Empty search dimension: {name}.{parameter}")
            for value in candidates:
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError(f"Nonfinite or nonnumeric grid value: {name}.{parameter}")
                if value < 0 or (parameter == "threshold" and value > 1):
                    raise ValueError(f"Invalid grid value: {name}.{parameter}={value}")
            dimensions.append((name, parameter, list(dict.fromkeys(candidates))))
    if not dimensions:
        raise ValueError("Search grid is empty")
    return config, dimensions


def _candidates(base, dimensions):
    for values in product(*(dimension[2] for dimension in dimensions)):
        candidate = deepcopy(base)
        for (name, parameter, _), value in zip(dimensions, values):
            candidate["postprocess"]["classes"][name][parameter] = value
        yield candidate


def search_probability_maps(probabilities, ground_truth, config=None, *, objective,
                            grid=None, method="staged", maximize=True):
    from terradelta.metrics import evaluate_predictions

    if not objective or not isinstance(objective, str):
        raise ValueError("An explicit evaluator objective key is required")
    if method not in {"cartesian", "staged"}:
        raise ValueError("Search method must be cartesian or staged")
    original_count = len(probabilities)
    probabilities = {sample_id(key): check_probabilities(value, context=str(key))
                     for key, value in probabilities.items()}
    if len(probabilities) != original_count:
        raise ValueError("Duplicate prediction IDs after string conversion")
    if not probabilities:
        raise ValueError("Probability maps are empty")
    ground_truth = [{**row, "id": sample_id(row["id"])} for row in ground_truth]
    identifiers = [sample_id(row["id"]) for row in ground_truth]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate ground-truth IDs")
    if set(identifiers) != set(probabilities):
        missing = sorted(set(identifiers) - set(probabilities))
        extra = sorted(set(probabilities) - set(identifiers))
        raise ValueError(f"Prediction/GT ID mismatch: missing predictions={missing}, extra predictions={extra}")
    base, dimensions = _prepare(config, DEFAULT_GRID if grid is None else grid)
    trials, cache = [], {}

    def evaluate(candidate, stage):
        key = json.dumps(candidate["postprocess"], sort_keys=True)
        if key not in cache:
            rows = [prediction_row(identifier, probabilities[identifier], candidate) for identifier in identifiers]
            scores = evaluate_predictions(rows, ground_truth)
            value = objective_value(scores, objective)
            trial = {"trial": len(trials), "stage": stage, "objective": objective, "objective_value": value,
                     **{f"{name}.{parameter}": candidate["postprocess"]["classes"][name][parameter]
                        for name in CLASSES for parameter in PARAMETERS},
                     "metrics": json.dumps(scores, sort_keys=True, allow_nan=False)}
            trials.append(trial)
            cache[key] = (value, dict(scores))
        return cache[key]

    best_config, best_value, best_scores = None, None, None

    def consider(candidate, stage):
        nonlocal best_config, best_value, best_scores
        value, scores = evaluate(candidate, stage)
        if best_value is None or (value > best_value if maximize else value < best_value):
            best_config, best_value, best_scores = deepcopy(candidate), value, scores

    if method == "cartesian":
        for candidate in _candidates(base, dimensions):
            consider(candidate, "cartesian")
    else:
        consider(base, "initial")
        for parameter in PARAMETERS:
            stage_dimensions = [dimension for dimension in dimensions if dimension[1] == parameter]
            if stage_dimensions:
                stage_base = deepcopy(best_config)
                for candidate in _candidates(stage_base, stage_dimensions):
                    consider(candidate, parameter)
    return {"best_config": best_config, "best_score": best_value, "best_scores": best_scores,
            "results": trials, "objective": objective, "method": method, "maximize": maximize}


def save_search_results(result, output_csv, best_yaml):
    output_csv, best_yaml = Path(output_csv), Path(best_yaml)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    best_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["results"][0]))
        writer.writeheader()
        writer.writerows(result["results"])
    best = deepcopy(result["best_config"])
    best["search_result"] = {key: result[key] for key in ("objective", "method", "maximize", "best_score")}
    with best_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best, handle, sort_keys=False)


def search_thresholds(pred_dir, ground_truth, config=None, *, objective, grid=None,
                      method="staged", maximize=True, ground_truth_kind="auto",
                      output_csv=None, best_yaml=None):
    if (output_csv is None) != (best_yaml is None):
        raise ValueError("Provide both output_csv and best_yaml, or neither")
    rows = load_ground_truth(ground_truth, kind=ground_truth_kind) if isinstance(ground_truth, (str, Path)) else ground_truth
    result = search_probability_maps(load_probability_maps(pred_dir), rows, config, objective=objective,
                                     grid=grid, method=method, maximize=maximize)
    if output_csv is not None:
        save_search_results(result, output_csv, best_yaml)
    return result
