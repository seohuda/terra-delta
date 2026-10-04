"""Small, resumable v2 calibration on real legacy and synthetic stress validation.

Polygonize each pixel threshold once, then reuse exact shape/presence statistics
for all presence thresholds. Both validation sets receive equal selection weight.
This module is evaluation-only and is excluded from submission packages.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.metrics.evaluation import _polygon_union, _shape_f1
from terradelta.postprocess.polygons import mask_to_polygons_reference
from terradelta.utils.io import atomic_json
from .v2 import CLASSES, V2Predictor

PRESENCE = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
PIXEL = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MODEL = {"architecture": "siamese_v2", "alignment": True, "alignment_scales": [3, 4], "max_offset": 1.5}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def statistics(gt, predicted, shape, nochange):
    tp = int(np.sum(gt & predicted))
    fp = int(np.sum(~gt & predicted))
    fn = int(np.sum(gt & ~predicted))
    tn = int(np.sum(~gt & ~predicted))

    def ratio(a, b):
        return float(a / b) if b else 0.0

    macro = (ratio(2 * tp, 2 * tp + fp + fn) + ratio(2 * tn, 2 * tn + fp + fn)) / 2
    score = ratio(float(np.sum(shape[gt & predicted])), int(gt.sum()))
    return {
        "presence_macro_f1": macro,
        "shape_score": score,
        "score": (macro + score) / 2,
        "gt_positive_count": int(gt.sum()),
        "presence_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "no_change_fp_count": int(np.sum(predicted & nochange)),
        "no_change_count": int(nochange.sum()),
    }


def cache_probabilities(checkpoint, manifest, output, device):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "checkpoint": digest(checkpoint),
        "manifest": digest(manifest),
        "model": MODEL,
        "preparation": "ImageNet_RGB_float32_pre_then_post_v1",
        "batch_size": 8,
    }
    data = IndependentChangeDataset(manifest)
    if output.exists():
        with np.load(output, allow_pickle=False) as z:
            if json.loads(str(z["identity"])) != identity:
                raise ValueError("Existing probability cache identity differs")
            pixels, presence = z["pixels"], z["presence"]
            if pixels.shape != (len(data), 2, 256, 256) or presence.shape != (len(data), 2):
                raise ValueError("Incomplete probability cache")
            return data, pixels, presence
    predictor = V2Predictor(checkpoint, {"model": MODEL}, device)
    pixels = np.empty((len(data), 2, 256, 256), np.float32)
    presence = np.empty((len(data), 2), np.float32)
    for start in range(0, len(data), 8):
        samples = [data[i] for i in range(start, min(start + 8, len(data)))]
        image = torch.stack([s["image"] for s in samples])
        p, q = predictor.probabilities(image)
        pixels[start : start + len(samples)] = p
        presence[start : start + len(samples)] = q
    tmp = output.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pixels=pixels, presence=presence, identity=json.dumps(identity, sort_keys=True))
    tmp.replace(output)
    return data, pixels, presence


def evaluate_grid(data, pixels, presence):
    gt_geometries = []
    for sample in data:
        if not sample["valid_mask"].all():
            raise ValueError("Calibration requires fully valid validation samples")
        gt_geometries.append(
            [
                _polygon_union(
                    mask_to_polygons_reference(
                        sample["target"][c].numpy() > 0, min_area=0, min_pos_area=0, simplify_px=0
                    ),
                    sample["id"],
                )
                for c in range(2)
            ]
        )
    gt = np.array([[not g.is_empty for g in pair] for pair in gt_geometries])
    nochange = ~gt.any(1)
    trials = {name: [] for name in CLASSES}
    for c, name in enumerate(CLASSES):
        for pixel in PIXEL:
            exists = np.zeros(len(data), bool)
            shape = np.zeros(len(data), float)
            for i in range(len(data)):
                mask = pixels[i, c] >= pixel
                geometry = _polygon_union(
                    mask_to_polygons_reference(
                        mask, min_area=30, min_pos_area=20, simplify_px=0.5, ndigits=2
                    ),
                    str(i),
                )
                if geometry.area >= 20:
                    exists[i] = True
                    if gt[i, c]:
                        shape[i] = _shape_f1(geometry, gt_geometries[i][c])
            for threshold in (*PRESENCE, None):
                flags = exists if threshold is None else exists & (presence[:, c] >= threshold)
                trials[name].append(
                    {
                        "presence": threshold,
                        "pixel": pixel,
                        "metrics": statistics(gt[:, c], flags, shape, nochange),
                    }
                )
        print(json.dumps({"event": "class_grid_complete", "class": name, "samples": len(data)}), flush=True)
    return trials


def select_plateau(trials):
    """Within 0.01 of equal-set maximum, prefer stable immediate neighbors."""
    for row in trials:
        row["utility"] = (row["legacy"]["score"] + row["stress"]["score"]) / 2
    maximum = max(r["utility"] for r in trials)
    index = {(r["presence"], r["pixel"]): r for r in trials}
    for row in trials:
        j = PIXEL.index(row["pixel"])
        pixel_neighbors = PIXEL[max(0, j - 1) : j + 2]
        if row["presence"] is None:
            presence_neighbors = (None,)
        else:
            k = PRESENCE.index(row["presence"])
            presence_neighbors = PRESENCE[max(0, k - 1) : k + 2]
        neighbors = [index[p, x]["utility"] for p in presence_neighbors for x in pixel_neighbors]
        row["neighbor_min"] = min(neighbors)
        row["neighbor_mean"] = float(np.mean(neighbors))
    candidates = [r for r in trials if r["utility"] >= maximum - 0.01]
    return max(
        candidates,
        key=lambda r: (
            r["neighbor_min"],
            r["neighbor_mean"],
            r["legacy"]["score"],
            -r["legacy"]["no_change_fp_count"],
            r["presence"] is None,
            -r["pixel"],
        ),
    )


def run(checkpoint_dir, legacy, stress, output, device="cuda"):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = {step: Path(checkpoint_dir) / f"step-{step:04d}.pt" for step in (50, 150, 250)}
    identity = {
        "checkpoint_sha256": {str(k): digest(v) for k, v in checkpoints.items()},
        "legacy_manifest_sha256": digest(legacy),
        "stress_manifest_sha256": digest(stress),
        "presence_grid": list(PRESENCE),
        "pixel_grid": list(PIXEL),
        "reference_polygonization": True,
        "calibration_code_sha256": digest(__file__),
        "schema_version": 1,
    }
    if output.exists():
        result = json.loads(output.read_text())
        if result.get("identity") != identity or result.get("status") != "complete":
            raise ValueError("Existing coarse result does not match this completed sweep")
        print("Reusing completed coarse sweep", flush=True)
        return result
    results = []
    started = time.monotonic()
    for step, checkpoint in checkpoints.items():
        part = output.parent / f"coarse-step-{step:04d}.json"
        if part.exists():
            item = json.loads(part.read_text())
            if item.get("identity") != identity:
                raise ValueError("Partial checkpoint sweep identity changed")
        else:
            sets = {}
            for name, manifest in [("legacy", legacy), ("stress", stress)]:
                data, p, q = cache_probabilities(
                    checkpoint, manifest, output.parent / f"probabilities-step-{step:04d}-{name}.npz", device
                )
                sets[name] = evaluate_grid(data, p, q)
                del p, q, data
            trials = {}
            selections = {}
            for name in CLASSES:
                combined = [
                    {
                        "presence": a["presence"],
                        "pixel": a["pixel"],
                        "legacy": a["metrics"],
                        "stress": b["metrics"],
                    }
                    for a, b in zip(sets["legacy"][name], sets["stress"][name])
                ]
                selections[name] = select_plateau(combined)
                trials[name] = combined
            item = {
                "identity": identity,
                "step": step,
                "checkpoint": str(checkpoint),
                "classes": trials,
                "selected_classes": selections,
                "balanced_score": float(np.mean([r["utility"] for r in selections.values()])),
            }
            atomic_json(part, item)
        results.append(item)
        print(
            json.dumps(
                {
                    "event": "checkpoint_complete",
                    "step": step,
                    "balanced_score": item["balanced_score"],
                    "selected": item["selected_classes"],
                }
            ),
            flush=True,
        )
    selected = max(
        results,
        key=lambda r: (
            r["balanced_score"],
            sum(x["neighbor_min"] for x in r["selected_classes"].values()),
            -r["step"],
        ),
    )
    result = {
        "status": "complete",
        "identity": identity,
        "selection_rule": "equal real-legacy/synthetic-stress class scores; 0.01 plateau, best neighboring worst score",
        "checkpoints": results,
        "selected_step": selected["step"],
        "selected_classes": selected["selected_classes"],
        "elapsed_seconds": time.monotonic() - started,
        "main_submissions": 0,
        "training_steps_executed": 0,
    }
    atomic_json(output, result)
    return result
