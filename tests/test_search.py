"""Forward-only integration checks for polygon validation and probability search."""

import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from shapely.geometry import Polygon
import torch
import yaml

from terradelta.metrics import evaluate_predictions
from terradelta.training.search import (
    load_ground_truth, load_probability_maps, objective_value,
    search_probability_maps, search_thresholds,
)
from terradelta.training.validation import ground_truth_row, prediction_row, validate_model


@pytest.fixture
def probability_case():
    """Two positive classes, a hard negative, and different useful thresholds."""
    probabilities, truth = {}, []
    for identifier, label in (("001", 1), ("002", 2), ("003", 0)):
        mask = np.zeros((16, 16), np.uint8)
        mask[1:7, 1:7] = label
        prob = np.zeros((3, 16, 16), np.float32)
        prob[0] = 1
        if label:
            value = 0.65 if label == 1 else 0.85
            prob[label, 1:7, 1:7] = value
            prob[0, 1:7, 1:7] = 1 - value
        else:
            prob[1, 1:7, 1:7], prob[0, 1:7, 1:7] = 0.45, 0.55
            prob[2, 9:15, 9:15], prob[0, 9:15, 9:15] = 0.65, 0.35
        probabilities[identifier] = prob
        truth.append(ground_truth_row({"id": identifier, "mask": mask}))
    config = {"postprocess": {"mode": "argmax", "classes": {
        name: {"threshold": 0.4, "min_area": 0, "min_pos_area": 0, "simplify_px": 0}
        for name in ("new_building", "tree_removal")
    }}, "inference": {"tta": ["identity"]}}
    grid = {"new_building": {"threshold": [0.4, 0.5, 0.8]},
            "tree_removal": {"threshold": [0.5, 0.7, 0.9]}}
    return probabilities, truth, config, grid


def test_cartesian_finds_independent_thresholds_and_preserves_config(probability_case):
    probabilities, truth, config, grid = probability_case
    original = deepcopy(config)
    result = search_probability_maps(probabilities, truth, config, objective="score", grid=grid, method="cartesian")
    assert result["best_score"] == pytest.approx(1)
    settings = result["best_config"]["postprocess"]
    assert settings["mode"] == "threshold"
    assert settings["classes"]["new_building"]["threshold"] == 0.4
    assert settings["classes"]["tree_removal"]["threshold"] == 0.7
    assert len(result["results"]) == 9
    assert config == original
    rows = [prediction_row(key, prob, result["best_config"]) for key, prob in probabilities.items()]
    assert result["best_scores"] == evaluate_predictions(rows, truth)


def test_staged_search_all_parameters_and_cartesian_agree(probability_case):
    probabilities, truth, config, _ = probability_case
    grid = {name: {"threshold": thresholds, "min_area": [0, 40],
                   "min_pos_area": [0, 40], "simplify_px": [0, 0.5]}
            for name, thresholds in (("new_building", [0.4, 0.5]), ("tree_removal", [0.5, 0.7]))}
    staged = search_probability_maps(probabilities, truth, config, objective="score", grid=grid, method="staged")
    full = search_probability_maps(probabilities, truth, config, objective="score", grid=grid, method="cartesian")
    assert staged["best_score"] == full["best_score"] == pytest.approx(1)
    assert len(staged["results"]) < len(full["results"]) == 256
    assert {row["stage"] for row in staged["results"]} == {
        "initial", "threshold", "min_area", "min_pos_area", "simplify_px"}
    assert max(row["objective_value"] for row in staged["results"]) == staged["best_score"]


def test_search_resolves_alias_and_direct_override_conflicts(probability_case):
    probabilities, truth, config, grid = probability_case
    config["postprocess"]["min_component_area"] = 0
    config["postprocess"]["new_building"] = {
        "threshold": 0.99, "min_area": 1000, "min_component_area": 0, "simplification_tolerance": 0,
        "morphology": {"opening": 0},
    }
    config["postprocess"]["tree_removal"] = {"threshold": 0.99, "min_positive_total_area": 1000}
    grid["new_building"]["min_area"] = [0]
    grid["tree_removal"]["min_pos_area"] = [0]
    result = search_probability_maps(probabilities, truth, config, objective="score", grid=grid, method="cartesian")
    assert result["best_score"] == pytest.approx(1)
    settings = result["best_config"]["postprocess"]
    assert "new_building" not in settings and "tree_removal" not in settings
    assert "min_component_area" not in settings
    assert "min_component_area" not in settings["classes"]["new_building"]
    assert settings["classes"]["new_building"]["morphology"] == {"opening": 0}


def test_objective_is_explicit_nested_finite_and_can_be_minimized(probability_case):
    probabilities, truth, config, grid = probability_case
    assert objective_value({"classes": {"new_building": {"score": 0.3}}}, "classes.new_building.score") == 0.3
    for objective, scores in (("missing", {"score": 1}), ("score", {"score": np.nan}),
                              ("score", {"score": {}}), ("score", {"score": True})):
        with pytest.raises(ValueError):
            objective_value(scores, objective)
    with pytest.raises(ValueError, match="explicit"):
        search_probability_maps(probabilities, truth, config, objective="", grid=grid)
    with pytest.raises(ValueError, match="absent"):
        search_probability_maps(probabilities, truth, config, objective="iou", grid=grid)
    result = search_probability_maps(probabilities, truth, config, objective="score", grid=grid,
                                     method="cartesian", maximize=False)
    assert result["best_score"] == min(row["objective_value"] for row in result["results"])


@pytest.mark.parametrize("grid", [
    {}, {"new_building": {"threshold": []}}, {"unknown": {"threshold": [0.5]}},
    {"new_building": {"threshold": [1.1]}}, {"new_building": {"min_area": [-1]}},
    {"new_building": {"threshold": [float("nan")]}}, {"new_building": {"threshold": "0.5"}},
    {"new_building": {"typo": [1]}},
])
def test_invalid_grids_fail(probability_case, grid):
    probabilities, truth, config, _ = probability_case
    with pytest.raises(ValueError):
        search_probability_maps(probabilities, truth, config, objective="score", grid=grid)


def test_id_alignment_is_strict(probability_case):
    probabilities, truth, config, grid = probability_case
    with pytest.raises(ValueError, match="mismatch"):
        search_probability_maps({"other": probabilities["001"]}, truth, config, objective="score", grid=grid)
    with pytest.raises(ValueError, match="Duplicate"):
        search_probability_maps(probabilities, truth + truth[:1], config, objective="score", grid=grid)


def _write_gt(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "new_building", "tree_removal"])
        writer.writeheader()
        writer.writerows(rows)


def test_probability_files_polygon_csv_and_result_artifacts(tmp_path, probability_case):
    probabilities, truth, config, grid = probability_case
    pred_dir = tmp_path / "probabilities"
    pred_dir.mkdir()
    for key, prob in probabilities.items():
        np.save(pred_dir / f"{key}.npy", prob)
    gt_csv = tmp_path / "gt.csv"
    _write_gt(gt_csv, list(reversed(truth)))
    assert load_ground_truth(gt_csv)[0]["id"] == "003"
    assert isinstance(load_probability_maps(pred_dir)["001"], np.ndarray)
    result = search_thresholds(pred_dir, gt_csv, config, objective="score", grid=grid,
                              output_csv=tmp_path / "out" / "trials.csv", best_yaml=tmp_path / "out" / "best.yaml")
    assert result["best_score"] == pytest.approx(1)
    with (tmp_path / "out" / "trials.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result["results"])
    assert "classes" in json.loads(rows[0]["metrics"])
    best = yaml.safe_load((tmp_path / "out" / "best.yaml").read_text())
    assert best["search_result"]["objective"] == "score"
    assert best["postprocess"]["mode"] == "threshold"
    assert best["inference"] == config["inference"]


@pytest.mark.parametrize("array", [np.zeros((2, 8, 8), np.float32), np.zeros((3, 0, 8), np.float32),
                                   np.full((3, 8, 8), np.nan, np.float32), np.ones((3, 8, 8), np.int32),
                                   np.full((3, 8, 8), 2, np.float32)])
def test_bad_probability_files_are_rejected(tmp_path, array):
    np.save(tmp_path / "001.npy", array)
    with pytest.raises(ValueError):
        load_probability_maps(tmp_path)


def test_object_probability_files_and_empty_directory_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="No .npy"):
        load_probability_maps(tmp_path)
    np.save(tmp_path / "001.npy", np.array([{"unsafe": "object"}], dtype=object))
    with pytest.raises(ValueError):
        load_probability_maps(tmp_path)


def _manifest_fixture(tmp_path):
    rows = []
    for index in range(3):
        image = np.full((16, 16, 3), 127, np.uint8)
        Image.fromarray(image).save(tmp_path / f"{index}_pre.png")
        Image.fromarray(image).save(tmp_path / f"{index}_post.png")
        mask = np.zeros((16, 16), np.uint8)
        mask[1:7, 1:7] = 255
        Image.fromarray(mask).save(tmp_path / f"{index}_building.png")
        rows.append({"id": f"00{index}", "pre": f"{index}_pre.png", "post": f"{index}_post.png",
                     "new_building": f"{index}_building.png", "tree_removal": "absent", "region_id": str(index),
                     "source": "synthetic", "license_status": "commercial_ok", "label_status": "reviewed"})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_ground_truth_manifest_reads_labels_and_relative_paths(tmp_path):
    manifest = _manifest_fixture(tmp_path)
    rows = load_ground_truth(manifest)
    assert [row["id"] for row in rows] == ["000", "001", "002"]
    assert all(row["tree_removal"] == "" for row in rows)
    assert Polygon(json.loads(rows[0]["new_building"])[0]).area == 36


class ForwardOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.child = torch.nn.Dropout()
        self.observations = []

    def forward(self, image):
        self.observations.append((self.training, torch.is_grad_enabled(), len(image)))
        return image[:, :3] + self.bias[None, :, None, None]


def _tensor_samples(probabilities, truth):
    # Log probability inputs exercise the real Predictor softmax and batching.
    samples = []
    for (identifier, prob), row in zip(probabilities.items(), truth):
        image = torch.zeros((6, *prob.shape[1:]))
        image[:3] = torch.log(torch.from_numpy(prob).clamp_min(1e-9))
        samples.append({**row, "id": identifier, "image": image})
    return samples


def test_validate_model_real_predictor_polygon_evaluator_saved_maps_and_modes(tmp_path, probability_case):
    probabilities, truth, config, _ = probability_case
    config["validation"] = {"pred_dir": str(tmp_path), "batch_size": 2}
    config["postprocess"]["mode"] = "threshold"
    config["postprocess"]["classes"]["new_building"]["threshold"] = 0.5
    config["postprocess"]["classes"]["tree_removal"]["threshold"] = 0.7
    model = ForwardOnlyModel().train()
    model.child.eval()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    samples = _tensor_samples(probabilities, truth)
    scores = validate_model(model, samples, config)
    assert scores["score"] == pytest.approx(1)
    assert model.training and not model.child.training
    assert model.observations == [(False, False, 2), (False, False, 1)]
    assert model.bias.grad is None
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())
    assert sorted(path.name for path in tmp_path.glob("*.npy")) == ["001.npy", "002.npy", "003.npy"]
    saved = load_probability_maps(tmp_path)
    for key, prob in probabilities.items():
        np.testing.assert_allclose(saved[key], prob, atol=1e-7)
    assert scores == evaluate_predictions([prediction_row(key, prob, config) for key, prob in saved.items()], truth)


def test_validate_model_restores_modes_after_duplicate_or_unlabeled_failure(probability_case):
    probabilities, truth, config, _ = probability_case
    samples = _tensor_samples(probabilities, truth)
    model = ForwardOnlyModel().train()
    model.child.eval()
    for broken, message in ((samples + samples[:1], "Duplicate"),
                            ([{**samples[0], "has_labels": False}], "annotated")):
        with pytest.raises(ValueError, match=message):
            validate_model(model, broken, config)
        assert model.training and not model.child.training
        assert model.bias.grad is None


def test_validation_uses_exported_exteriors_not_pixel_overlap():
    mask = np.zeros((16, 16), np.uint8)
    mask[1:9, 1:9] = 1
    mask[3:7, 3:7] = 0
    row = ground_truth_row({"id": "donut", "mask": mask})
    assert (mask == 1).sum() == 48
    assert Polygon(json.loads(row["new_building"])[0]).area == 64
    small = np.zeros((16, 16), np.uint8)
    small[0, 0] = 1
    assert ground_truth_row({"id": "tiny", "mask": small})["new_building"] != ""


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_search_cli_requires_objective_and_writes_outputs(tmp_path, probability_case):
    probabilities, truth, config, grid = probability_case
    for key, prob in probabilities.items():
        np.save(tmp_path / f"{key}.npy", prob)
    gt_csv = tmp_path / "gt.csv"
    _write_gt(gt_csv, truth)
    config["search"] = {"grid": grid}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    script = _script("search_thresholds.py")
    common = ["--pred-dir", str(tmp_path), "--ground-truth", str(gt_csv), "--config", str(config_path)]
    with pytest.raises(SystemExit):
        script.main(common)
    result = script.main(common + ["--objective", "score", "--output-csv", str(tmp_path / "search.csv"),
                                   "--best-yaml", str(tmp_path / "best.yaml")])
    assert result["best_score"] == pytest.approx(1)
    assert (tmp_path / "search.csv").exists() and (tmp_path / "best.yaml").exists()


def test_validate_cli_uses_parent_split_and_offline_checkpoint_factory(tmp_path, monkeypatch):
    manifest = _manifest_fixture(tmp_path)
    config = {"model": {"encoder": "resnet18", "encoder_weights": "imagenet"},
              "data": {"manifest": str(manifest), "split": {"strategy": "region", "val_fraction": .2, "seed": 0}}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    script = _script("validate.py")
    model = ForwardOnlyModel()
    built, loaded = [], []

    def build(config):
        built.append(config)
        return model

    monkeypatch.setattr(script, "build_model", build)
    monkeypatch.setattr(script, "load_checkpoint", lambda path, target: loaded.append((path, target)))
    scores = script.main(["--config", str(config_path), "--checkpoint", "existing.pt",
                          "--output", str(tmp_path / "validation.json")])
    assert scores["num_samples"] == 1
    assert loaded == [("existing.pt", model)]
    assert built[0]["model"]["encoder_weights"] is None
    assert json.loads((tmp_path / "validation.json").read_text()) == scores
    assert config["model"]["encoder_weights"] == "imagenet"
