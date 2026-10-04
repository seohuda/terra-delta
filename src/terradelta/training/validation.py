"""Validation through the same exterior polygons used for submission.

``validate_model`` receives an already split dataset. It does not select a split
or compute a pixel proxy. Ground-truth masks are converted without prediction
area filtering or simplification, then scored by the local polygon evaluator.
``validation.pred_dir`` optionally saves one ``(3, H, W)`` .npy per sample.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


CLASSES = ("new_building", "tree_removal")


def sample_id(value):
    """Keep IDs lossless (including leading zeros) and safe as .npy filenames."""
    value = str(value)
    if not value.strip() or value in {".", ".."} or any(c in value for c in "/\\\0"):
        raise ValueError(f"Invalid sample ID: {value!r}")
    return value


def check_probabilities(probabilities, *, context="prediction"):
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3 or probabilities.shape[0] != 3 or min(probabilities.shape[1:]) < 1:
        raise ValueError(f"{context}: expected probabilities with shape (3, H, W)")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise ValueError(f"{context}: probabilities must have a floating dtype")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"{context}: probabilities must be finite and in [0, 1]")
    return probabilities


def prediction_row(identifier, probabilities, config):
    from terradelta.postprocess import predictions_to_polygons, serialize_polygons

    polygons = predictions_to_polygons(check_probabilities(probabilities, context=identifier), config)
    return {"id": sample_id(identifier), **{name: serialize_polygons(polygons[name]) for name in CLASSES}}


def ground_truth_row(sample):
    """Accept polygon labels or an integer mask; never threshold GT probabilities."""
    from terradelta.postprocess import mask_to_polygons_reference, serialize_polygons

    identifier = sample_id(sample["id"])
    if all(name in sample for name in CLASSES):
        return {
            "id": identifier,
            **{name: sample[name] if isinstance(sample[name], str) else serialize_polygons(sample[name])
               for name in CLASSES},
        }
    if "mask" not in sample:
        raise ValueError(f"{identifier}: validation requires mask or polygon ground truth")
    mask = sample["mask"]
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError(f"{identifier}: expected a HxW label mask containing only 0, 1, 2")
    return {
        "id": identifier,
        **{name: mask_to_polygons_reference(
            mask == index, min_area=0, min_pos_area=0, simplify_px=0,
        ) for index, name in enumerate(CLASSES, start=1)},
    }


def validate_model(model, dataset, config, device="cpu"):
    """Return ``evaluate_predictions(pred_rows, gt_rows)`` unchanged.

    All model modes are restored even if prediction/evaluation fails. Inference
    uses no gradients or optimizer. IDs must be unique within the validation set.
    """
    from terradelta.inference.predictor import Predictor
    from terradelta.metrics import evaluate_predictions

    options = config.get("validation", {})
    batch_size = int(options.get("batch_size", config.get("training", {}).get("validation_batch_size",
                     config.get("training", {}).get("batch_size", config.get("inference", {}).get("batch_size", 16)))))
    if batch_size < 1:
        raise ValueError("validation.batch_size must be positive")
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty")
    pred_dir = options.get("pred_dir")
    if pred_dir:
        pred_dir = Path(pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=int(options.get("num_workers", 0)), collate_fn=list)
    modes = [(module, module.training) for module in model.modules()]
    pred_rows, gt_rows, seen = [], [], set()
    try:
        model.eval()
        prediction_config = deepcopy(config)
        if prediction_config.get("inference", {}).get("tta") == []:
            prediction_config["inference"]["tta"] = ["identity"]
        predictor = Predictor(model, prediction_config, device)
        with torch.inference_mode():
            for samples in loader:
                images = torch.stack([sample["image"] for sample in samples])
                probabilities = np.asarray(predictor.predict_batch(images))
                if probabilities.ndim != 4 or probabilities.shape[:2] != (len(samples), 3):
                    raise ValueError("Predictor must return numpy probabilities with shape (B, 3, H, W)")
                if probabilities.shape[2:] != tuple(images.shape[-2:]):
                    raise ValueError("Predictor spatial shape does not match validation images")
                for sample, prob in zip(samples, probabilities):
                    if "valid_mask" in sample and not bool(sample["valid_mask"].all()):
                        raise ValueError("Validation requires fully annotated tiles; partial weak labels belong in training only")
                    if not sample.get("has_labels", True):
                        raise ValueError("Validation requires annotated ground truth, not inference-only samples")
                    identifier = sample_id(sample["id"])
                    if identifier in seen:
                        raise ValueError(f"Duplicate validation ID: {identifier}")
                    seen.add(identifier)
                    gt_row = ground_truth_row(sample)
                    if "mask" in sample and tuple(sample["mask"].shape) != tuple(prob.shape[1:]):
                        raise ValueError(f"{identifier}: ground-truth shape does not match probabilities")
                    pred_rows.append(prediction_row(identifier, prob, config))
                    gt_rows.append(gt_row)
                    if pred_dir:
                        np.save(pred_dir / f"{identifier}.npy", prob, allow_pickle=False)
        scores = evaluate_predictions(pred_rows, gt_rows)
        if not isinstance(scores, Mapping):
            raise TypeError("Polygon evaluator must return a score dictionary")
        return dict(scores)
    finally:
        for module, training in modes:
            module.training = training
