"""Inference-only component vetoes; identity polygons are the sole shape source.

Matching is fixed before validation: same-class, one-to-one, IoU >= .2 OR
centroid distance <= 4 pixels with area ratio in [1/4, 4]. Four-connectivity
and polygon ordering come from the existing official reference conversion.
"""

import json
from collections.abc import Mapping

import numpy as np
from shapely import contains_xy
import torch

from terradelta.postprocess.polygons import reference_components, reference_exteriors, serialize_polygons
from terradelta.utils.io import write_prediction_csv
from .predictor import discover_pairs, read_image
from .v2 import CLASSES, V2Predictor, class_options, prepare_v2_pair

TRANSFORMS = ("identity", "hflip", "vflip", "rot180")
RULES = {
    "A": {"persistence_min": 3, "mean_iou_min": 0.0},
    "B": {"persistence_min": 4, "mean_iou_min": 0.0},
    "C": {"persistence_min": 3, "mean_iou_min": 0.2},
    "D": {"persistence_min": 3, "mean_iou_min": 0.3},
    "E": {"persistence_min": 3, "mean_iou_min": 0.4},
}


def transform(value, name):
    """Apply the paired spatial transform; each supported transform is its inverse."""
    if name not in TRANSFORMS:
        raise ValueError("Unknown geometric transform")
    if value.ndim < 2:
        raise ValueError("Transform needs two spatial dimensions")
    axes = {"identity": (), "hflip": (-1,), "vflip": (-2,), "rot180": (-2, -1)}[name]
    if not axes:
        return value
    if isinstance(value, torch.Tensor):
        return torch.flip(value, axes)
    return np.flip(value, axes).copy()


def validate_stability(config):
    if not isinstance(config, Mapping) or set(config) - {"enabled", "classes", "appearance_consistency"}:
        raise ValueError("Invalid stability configuration")
    for name in ("enabled", "appearance_consistency"):
        if name in config and not isinstance(config[name], bool):
            raise ValueError(f"{name} must be boolean")
    classes = config.get("classes", {})
    if not isinstance(classes, Mapping) or set(classes) - set(CLASSES):
        raise ValueError("Stability classes must be independent building/tree rules")
    for rule in classes.values():
        if not isinstance(rule, Mapping) or set(rule) != {"persistence_min", "mean_iou_min"}:
            raise ValueError("Expected persistence_min and mean_iou_min")
        count, iou = rule["persistence_min"], rule["mean_iou_min"]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4:
            raise ValueError("persistence_min must be an integer in [1,4]")
        if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not np.isfinite(iou) or not 0 <= iou <= 1:
            raise ValueError("mean_iou_min must be finite in [0,1]")
    return config


def match_components(identity, auxiliary):
    """Deterministic one-to-one matching of unsimplified same-class components.

    Prefer IoU, then centroid distance, then reference indices. A split or merge
    cannot give several identity components credit for the same auxiliary part.
    """
    edges = []
    for i, original in enumerate(identity):
        for j, other in enumerate(auxiliary):
            intersection = original.intersection(other).area
            iou = intersection / (original.area + other.area - intersection)
            drift = original.centroid.distance(other.centroid)
            ratio = other.area / original.area
            if iou >= 0.2 or (drift <= 4.0 and 0.25 <= ratio <= 4.0):
                edges.append((-iou, drift, i, j))
    used_i, used_j, matches = set(), set(), {}
    for neg_iou, drift, i, j in sorted(edges):
        if i not in used_i and j not in used_j:
            matches[i] = {"index": j, "iou": -neg_iou, "centroid_drift": drift}
            used_i.add(i)
            used_j.add(j)
    return matches


def _parts(probability, presence, options):
    threshold = options["presence_threshold"]
    if threshold is not None and presence < threshold:
        return []
    parts = [p for p in reference_components(probability >= options["pixel_threshold"])
             if p.area >= options["min_area"]]
    return parts if sum(p.area for p in parts) >= options["min_pos_area"] else []


def _component_pixels(part):
    xmin, ymin, xmax, ymax = (int(v) for v in part.bounds)
    y, x = np.mgrid[ymin:ymax, xmin:xmax]
    inside = contains_xy(part, x + 0.5, y + 0.5)
    return y[inside], x[inside]


def component_features(pixels, presence, config, appearance=None):
    """Features from inverse-restored maps, never from a mean-probability mask.

    Persistence/area moments include identity. IoU summaries use the three
    auxiliary views, with zero for missing matches; drift uses their matched
    views only. Probability std is mean pixelwise
    std over the identity component, and consensus is its fraction supported by
    >=3 views. largest_confidence is identity mean confidence of the largest
    eligible component of that class. Geometry/pixel area retains holes.
    """
    pixels, presence = np.asarray(pixels), np.asarray(presence)
    if pixels.shape != (4, 2, 256, 256) or presence.shape != (4, 2):
        raise ValueError("Expected four restored 2x256x256 maps and presence pairs")
    for values in (pixels, presence):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("TTA probabilities must be finite in [0,1]")
    if appearance is not None:
        ap, aq = (np.asarray(v) for v in appearance)
        if ap.shape != (2, 256, 256) or aq.shape != (2,):
            raise ValueError("Invalid appearance probability shapes")
        if any(not np.isfinite(v).all() or ((v < 0) | (v > 1)).any() for v in (ap, aq)):
            raise ValueError("Appearance probabilities must be finite in [0,1]")
    result = {}
    for channel, name in enumerate(CLASSES):
        options = class_options(config, name)
        views = [_parts(p[channel], q[channel], options) for p, q in zip(pixels, presence)]
        originals = views[0]
        polygons = reference_exteriors(originals, options["min_area"], options["min_pos_area"],
                                       options["simplify_px"], options["ndigits"])
        if len(polygons) != len(originals):
            raise ValueError("Identity component/exterior contract differs")
        masks = [_component_pixels(p) for p in originals]
        largest = max(range(len(originals)), key=lambda i: originals[i].area) if originals else None
        largest_confidence = float(pixels[0, channel][masks[largest]].mean()) if largest is not None else 0.0
        matches = [match_components(originals, parts) for parts in views[1:]]
        appearance_matches = (match_components(originals, _parts(ap[channel], aq[channel], options))
                              if appearance is not None else None)
        records = []
        for i, (part, polygon, positions) in enumerate(zip(originals, polygons, masks)):
            ious, areas, drift = [1.0], [part.area], [0.0]
            match_indices = [i]
            for view, match in zip(views[1:], matches):
                found = match.get(i)
                ious.append(found["iou"] if found else 0.0)
                match_indices.append(found["index"] if found else None)
                if found:
                    areas.append(view[found["index"]].area)
                    drift.append(found["centroid_drift"])
            values = pixels[:, channel][:, positions[0], positions[1]]
            persistence = sum(index is not None for index in match_indices)
            records.append({
                "polygon": polygon, "component_index": i,
                "original_mean_probability": float(values[0].mean()),
                "original_max_probability": float(values[0].max()), "component_area": float(part.area),
                "largest_confidence": largest_confidence,
                "persistence_count": persistence, "persistence_fraction": persistence / 4,
                "mean_matching_iou": float(np.mean(ious[1:])), "minimum_matching_iou": min(ious[1:]),
                "matching_ious": ious, "matching_indices": match_indices,
                "matching_component_area_mean": float(np.mean(areas)), "area_std": float(np.std(areas)),
                "area_coefficient_of_variation": float(np.std(areas) / np.mean(areas)),
                "centroid_drift": float(np.mean(drift[1:])) if len(drift) > 1 else 0.0,
                "maximum_centroid_drift": max(drift),
                "tta_probability_mean": float(values.mean()), "tta_probability_std": float(values.std(axis=0).mean()),
                "consensus_fraction": float(((values >= options["pixel_threshold"]).sum(0) >= 3).mean()),
                "appearance_matched": i in appearance_matches if appearance_matches is not None else None,
            })
        result[name] = records
    return result


def apply_stability(row, features, stability):
    """Remove rejected components from the existing identity serialization only."""
    stability = validate_stability(stability)
    result = dict(row)
    if not stability.get("enabled", False):
        return result
    for name in CLASSES:
        if not row[name]:
            continue  # The original verifier/presence veto always wins.
        records = features[name]
        original = json.loads(row[name])
        if original != [r["polygon"] for r in records]:
            raise ValueError("Stability features differ from identity polygon source")
        rule = stability.get("classes", {}).get(name, RULES["A"])
        kept = []
        for record in records:
            accepted = (record["persistence_count"] >= rule["persistence_min"]
                        and record["mean_matching_iou"] >= rule["mean_iou_min"])
            if stability.get("appearance_consistency", False):
                if record["appearance_matched"] is None:
                    raise ValueError("Appearance gate requires auxiliary inference")
                accepted = accepted and record["appearance_matched"]
            if accepted:
                kept.append(record["polygon"])
        if len(kept) != len(original):
            result[name] = serialize_polygons(kept)
    return result


def radiometric_pair(image):
    """One optional auxiliary arm: align PRE channel mean/std to POST; no resampling."""
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    before, after = (image[:, :3] * std + mean).clamp(0, 1), (image[:, 3:] * std + mean).clamp(0, 1)
    bmean, amean = before.mean((-2, -1), keepdim=True), after.mean((-2, -1), keepdim=True)
    ratio = (after.std((-2, -1), correction=0, keepdim=True)
             / before.std((-2, -1), correction=0, keepdim=True).clamp_min(.02)).clamp(.5, 2)
    corrected = ((before - bmean) * ratio + amean).clamp(0, 1)
    return torch.cat(((corrected - mean) / std, image[:, 3:]), dim=1)


@torch.inference_mode()
def tta_probabilities(predictor, image, identity=None):
    """Four separate frozen forwards. Return maps in original coordinates."""
    if identity is None:
        identity = predictor.probabilities(image)
    maps, presence = [identity[0]], [identity[1]]
    for name in TRANSFORMS[1:]:
        p, q = predictor.probabilities(transform(image, name))
        maps.append(transform(p, name))
        presence.append(q)
    return np.stack(maps), np.stack(presence)


class V23Predictor(V2Predictor):
    def __init__(self, checkpoint, config, device="cpu", model=None):
        self.stability = validate_stability(config.get("stability", {"enabled": False}))
        super().__init__(checkpoint, config, device, model)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def output_rows(self, identifiers, image, pixels, presence):
        rows = super().output_rows(identifiers, image, pixels, presence)
        if not self.stability.get("enabled", False) or not any(row[c] for row in rows for c in CLASSES):
            return rows
        maps, heads = tta_probabilities(self, image, (pixels, presence))
        appearance = (self.probabilities(radiometric_pair(image))
                      if self.stability.get("appearance_consistency", False) else None)
        return [apply_stability(row, component_features(maps[:, i], heads[:, i], self.config,
                (appearance[0][i], appearance[1][i]) if appearance is not None else None), self.stability)
                for i, row in enumerate(rows)]


def predict_directory(input_dir, output_path, checkpoint, config, device="cpu"):
    stability = validate_stability(config.get("stability", {"enabled": False}))
    if not stability.get("enabled", False):
        from .v2 import predict_directory as predict_v22
        return predict_v22(input_dir, output_path, checkpoint, config, device)
    root, ids = discover_pairs(input_dir)
    predictor = V23Predictor(checkpoint, config, device)
    batch_size = config.get("inference", {}).get("batch_size", 8)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    rows = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        image = torch.stack([prepare_v2_pair(read_image(root / "images" / i / "pre.png"),
                                            read_image(root / "images" / i / "post.png")) for i in batch_ids])
        pixels, presence = predictor.probabilities(image)
        rows.extend(predictor.output_rows(batch_ids, image, pixels, presence))
        print(f"Predicted {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
    write_prediction_csv(output_path, rows)
    return rows
