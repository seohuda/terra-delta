"""Exact exported-polygon validation and conservative v2.1 promotion gates."""

import torch

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.metrics.evaluation import _polygon_union, evaluate_predictions
from terradelta.postprocess.polygons import mask_to_polygons_reference
from terradelta.utils.io import write_prediction_csv
from .v2 import CLASSES, V2Predictor


@torch.inference_mode()
def evaluate_model(model, manifest, config, device, prediction_path=None):
    data = IndependentChangeDataset(manifest)
    predictor = V2Predictor(None, config, device, model=model)
    predictions, truth = [], []
    batch_size = config.get("inference", {}).get("batch_size", 8)
    for start in range(0, len(data), batch_size):
        samples = [data[i] for i in range(start, min(start + batch_size, len(data)))]
        if any(not s["valid_mask"].all() for s in samples):
            raise ValueError("Evaluation requires full-tile validation labels")
        images = torch.stack([s["image"] for s in samples])
        pixels, presence = predictor.probabilities(images)
        predictions.extend(predictor.output_rows([s["id"] for s in samples], images, pixels, presence))
        truth.extend({
            "id": s["id"], **{
                c: mask_to_polygons_reference(s["target"][i].numpy() > 0, 0, 0, 0)
                for i, c in enumerate(CLASSES)
            },
        } for s in samples)
    result = evaluate_predictions(predictions, truth)
    negatives = [i for i, row in enumerate(truth) if all(not row[c] for c in CLASSES)]
    result["no_change_count"] = len(negatives)
    result["no_change_fp_any"] = sum(
        any(_polygon_union(predictions[i][c], "prediction").area >= 20 for c in CLASSES)
        for i in negatives
    )
    for c in CLASSES:
        result["classes"][c]["no_change_fp_count"] = sum(
            _polygon_union(predictions[i][c], "prediction").area >= 20 for i in negatives
        )
    if prediction_path:
        write_prediction_csv(prediction_path, predictions)
    return result


def promotion_gate(candidate, baseline):
    """Compare identical fixed inference settings; forbid negative-only collapse."""
    reasons = []
    for name in ("legacy", "stress"):
        new, old = candidate[name], baseline[name]
        if new["num_samples"] != old["num_samples"] or new["no_change_count"] != old["no_change_count"]:
            raise ValueError("Validation populations differ")
        if new["score"] + 1e-10 < old["score"]:
            reasons.append(f"{name}: overall score regressed")
        if new["no_change_fp_any"] > old["no_change_fp_any"]:
            reasons.append(f"{name}: no-change false positives increased")
        for c in CLASSES:
            a, b = new["classes"][c], old["classes"][c]
            if a["gt_positive_count"] != b["gt_positive_count"]:
                raise ValueError("Validation positive populations differ")
            if a["presence_confusion"]["tp"] < b["presence_confusion"]["tp"]:
                reasons.append(f"{name}/{c}: positive recall regressed")
            if a["shape_score"] + 1e-10 < .95 * b["shape_score"]:
                reasons.append(f"{name}/{c}: shape score dropped more than 5 percent")
    if candidate["legacy"]["no_change_fp_any"] >= baseline["legacy"]["no_change_fp_any"]:
        reasons.append("legacy: real no-change false positives did not decrease")
    return {"passed": not reasons, "reasons": reasons}
