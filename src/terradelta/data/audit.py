"""Offline, bounded dataset evidence. A mock metric never establishes readiness."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from terradelta.data.dataset import ChangeDataset, MASK_CLASSES, read_manifest
from terradelta.data.pairing import training_eligibility, write_manifest
from terradelta.data.split import spatial_groups, split_manifest
from terradelta.inference.alignment import estimate_translation
from terradelta.utils.io import atomic_json

PATH_FIELDS = ("pre", "post", *MASK_CLASSES, "valid_mask", "review_mask")
RASTER_FIELDS = ("source_raster", "pre_source_raster", "post_source_raster", "pre_raster", "post_raster")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory(root):
    """Scan this repository only; deduplicate references across split manifests.

    No external paths, archives, virtual environments or Git objects are scanned.
    Physical file sizes count each path once, regardless of manifest references.
    """
    root = Path(root).resolve()
    manifests, folders, assets = [], [], set()
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [n for n in names if n not in {".git", ".venv", "__pycache__", ".pytest_cache",
                    ".ruff_cache", "data_audit"} and not (Path(directory) / n).is_symlink()]
        folder = Path(directory)
        if {"pre.png", "post.png"} <= set(files):
            folders.append(str(folder.relative_to(root)))
            assets.update(folder / f"{key}.png" for key in PATH_FIELDS if (folder / f"{key}.png").is_file())
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            path = folder / filename
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not {"id", "pre", "post"} <= set(reader.fieldnames or ()):
                    continue
                rows = list(reader)
            manifests.append({"path": str(path.relative_to(root)), "rows": len(rows), "bytes": path.stat().st_size})
            for row in rows:
                for key in PATH_FIELDS:
                    value = row.get(key)
                    if value and value.lower() != "absent":
                        resolved = (path.parent / value).resolve()
                        if resolved.is_relative_to(root) and resolved.is_file():
                            assets.add(resolved)
    asset_bytes = sum(p.stat().st_size for p in assets)
    return {"scan_root": str(root), "manifests": manifests, "sample_folders": sorted(folders),
            "unique_sample_folders": len(folders), "image_mask_files": len(assets),
            "image_mask_bytes": asset_bytes, "manifest_bytes": sum(m["bytes"] for m in manifests),
            "average_image_mask_bytes_per_sample": asset_bytes / len(folders) if folders else None,
            "scope_note": "Local repository only; split references are not additional samples; archives not expanded."}


def is_mock(row):
    return any(token in str(row.get("source", "")).lower().split("_") for token in ("mock", "demo", "fixture", "test"))


def provenance(row):
    """Do not promote self-asserted license strings or artificial geography."""
    required = ("source", "pre_source", "post_source", "mask_source", "region_id", "year_pre", "year_post",
                "resolution_m", "license_source", "source_url", "label_status", "reviewer")
    missing = [key for key in required if not str(row.get(key, "")).strip()]
    if not (row.get("bounds") and row.get("crs") or row.get("aoi_id") or row.get("location_id")):
        missing.append("geographic_evidence")
    reasons = ["missing " + ", ".join(missing)] if missing else []
    if is_mock(row):
        reasons.append("mock imagery/labels; dates and regions are invented")
    allowed, reason = training_eligibility(row)
    if not allowed:
        reasons.append(reason)
    if row.get("label_status") not in {"reviewed", "verified"}:
        reasons.append("labels not explicitly reviewed")
    if row.get("provenance_status") != "verified" or row.get("license_review_status") != "verified":
        reasons.append("documented provenance/license review not verified")
    try:
        resolution = float(row["resolution_m"])
        if int(row["year_pre"]) >= int(row["year_post"]) or not np.isfinite(resolution) or resolution <= 0:
            reasons.append("invalid acquisition interval/resolution")
    except (KeyError, TypeError, ValueError):
        reasons.append("acquisition interval/resolution unverified")
    return {**row, "synthetic": str(row.get("synthetic", "")).lower() in {"true", "1"} or "synthetic" in row.get("source", "").lower(),
            "mock": is_mock(row), "audit_status": "unverified" if reasons else "verified",
            "audit_training_eligible": not reasons, "audit_reason": "; ".join(reasons)}


def image_fingerprint(array):
    return hashlib.sha256(str((array.shape, array.dtype)).encode() + array.tobytes()).hexdigest()


def inspect_sample(row):
    """Strict original-channel checks precede the loader's RGB conversion."""
    issues, raw = [], {}
    for key in PATH_FIELDS:
        value = row.get(key)
        if not value or value.lower() == "absent":
            continue
        try:
            with Image.open(value) as image:
                array = np.array(image)
            if not np.isfinite(array).all():
                issues.append(("error", key + ": NaN/Inf"))
            if key in {"pre", "post"}:
                if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                    issues.append(("error", key + ": expected uint8 RGB, no implicit conversion"))
                if array.shape[:2] != (256, 256):
                    issues.append(("error", key + ": expected baseline dimensions 256x256"))
                if array.std() < 1:
                    issues.append(("warning", key + ": blank/nearly blank imagery"))
                if array.mean() < 8 or array.mean() > 247:
                    issues.append(("warning", key + ": extremely dark/bright imagery"))
            raw[key] = array
        except (OSError, ValueError, SyntaxError) as error:
            issues.append(("error", key + ": unreadable: " + str(error)))
    sample = None
    try:
        # Reuse canonical mask value, overlap, shape and review-scope validation.
        sample = ChangeDataset([row])[0]
    except (OSError, ValueError, SyntaxError) as error:
        issues.append(("error", "labels/pair: " + str(error)))
    mask = sample["mask"].numpy() if sample else None
    if mask is not None:
        for index, name in enumerate(MASK_CLASSES, 1):
            if str(row.get(name + "_present", "")).lower() in {"true", "1", "yes"} and not (mask == index).any():
                issues.append(("error", name + ": declared positive but empty mask"))
        if np.any(mask == -100):
            issues.append(("warning", "partial review scope; excluded from validation"))
        if not np.any(mask >= 0):
            issues.append(("error", "no reviewed label pixels"))
    identical = "pre" in raw and "post" in raw and np.array_equal(raw["pre"], raw["post"])
    if identical:
        issues.append(("info", "identical pre/post; legitimate no-change only when reviewed"))
    estimate = None
    if all(key in raw for key in ("pre", "post")):
        try:
            estimate = estimate_translation(raw["pre"], raw["post"])
            if not estimate["reliable"]:
                issues.append(("warning", "low phase correlation response"))
            elif estimate["magnitude"] > 4:
                issues.append(("warning", "alignment shift >4 pixels"))
        except (ValueError, cv2.error) as error:
            issues.append(("warning", "alignment unavailable: " + str(error)))
    synthetic = {}
    if mask is not None and provenance(row)["synthetic"] and "pre" in raw and "post" in raw and raw["pre"].shape == raw["post"].shape:
        positive = mask > 0
        if positive.any():
            difference = np.abs(raw["pre"].astype(float) - raw["post"].astype(float)).mean(axis=2)
            synthetic = {"changed_pixel_fraction_inside_gt": float((difference[positive] > 0).mean()),
                         "changed_pixel_fraction_outside_gt": float((difference[~positive] > 0).mean()) if (~positive).any() else None,
                         "pre_unique_colors_inside_gt": int(len(np.unique(raw["pre"][positive], axis=0))),
                         "post_unique_colors_inside_gt": int(len(np.unique(raw["post"][positive], axis=0))),
                         "note": "Heuristic diagnostics; no automatic claim of realistic synthesis."}
            if min(synthetic["pre_unique_colors_inside_gt"], synthetic["post_unique_colors_inside_gt"]) <= 1:
                issues.append(("warning", "synthetic positive region has constant-color fill"))
    return raw, mask, issues, estimate, synthetic


def distribution(masks):
    counts = {"background_only": 0, "new_building_only": 0, "tree_removal_only": 0, "both_classes": 0}
    pixels, samples, empty = np.zeros(3, dtype=np.int64), [], {name: 0 for name in MASK_CLASSES}
    for identifier, mask in masks.items():
        building, tree = (mask == 1), (mask == 2)
        category = "both_classes" if building.any() and tree.any() else "new_building_only" if building.any() else "tree_removal_only" if tree.any() else "background_only"
        counts[category] += 1
        row = {"id": identifier, "category": category, "valid_pixels": int((mask >= 0).sum()),
               "ignored_pixels": int((mask == -100).sum())}
        pixels += [int((mask == index).sum()) for index in range(3)]
        for name, binary in zip(MASK_CLASSES, (building, tree)):
            row[name + "_pixels"] = int(binary.sum())
            row[name + "_components"] = int(cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)[0] - 1)
            empty[name] += int(not binary.any())
        samples.append(row)
    total = int(pixels.sum())
    return {"valid_samples": len(masks), **counts, "empty_change_masks": counts["background_only"],
            "empty_binary_masks": empty, "pixel_count": dict(zip(("background", *MASK_CLASSES), map(int, pixels))),
            "pixel_ratio": dict(zip(("background", *MASK_CLASSES), (pixels / total).tolist() if total else [None] * 3)),
            "valid_pixel_count": total, "samples": samples, "component_connectivity": 8}


def leakage(rows, partitions):
    """Canonical spatial groups plus cross-partition decoded-image duplicates.

    Missing location evidence is unassessable, never reported as leakage-free.
    Group candidates include temporal chains regardless of acquisition interval.
    """
    candidates, temporal, errors = [], [], []
    try:
        groups = spatial_groups(rows)
        for group in groups:
            train = [rows[i] for i in group if partitions.get(rows[i]["id"]) == "train"]
            val = [rows[i] for i in group if partitions.get(rows[i]["id"]) == "val"]
            for a in train:
                for b in val:
                    record = {"train_id": a["id"], "val_id": b["id"], "reason": "shared spatial/source group"}
                    candidates.append(record)
                    if all(r.get("year_pre") and r.get("year_post") for r in (a, b)):
                        temporal.append({**record, "train_interval": f"{a['year_pre']}->{a['year_post']}",
                                         "val_interval": f"{b['year_pre']}->{b['year_post']}"})
    except ValueError as error:
        groups = []
        errors.append(str(error))
    verified_locations = sum(bool(not is_mock(r) and r.get("bounds") and r.get("crs")) for r in rows)
    return {"groups": len(groups), "spatial_candidates": candidates, "temporal_candidates": temporal,
            "errors": errors, "rows_with_real_coordinate_evidence": verified_locations,
            "status": "leakage_detected" if candidates else "unverified" if errors or verified_locations != len(rows) else "no_candidates",
            "note": "No candidates among placeholder region IDs is not evidence of real geographic separation."}


def montage(rows, masks, output, selections, predictions=None):
    """At most eight rows, 640 pixels wide; load only selected local samples."""
    chosen = selections[:8]
    if not chosen:
        return None
    canvas = Image.new("RGB", (640, 186 * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    lookup = {row["id"]: row for row in rows}
    for i, (category, identifier) in enumerate(chosen):
        row = lookup[identifier]
        with Image.open(row["pre"]) as image:
            pre = image.convert("RGB").resize((160, 160))
        with Image.open(row["post"]) as image:
            post = image.convert("RGB").resize((160, 160))
        mask = masks[identifier]
        color = np.zeros((*mask.shape, 3), np.uint8)
        color[mask == 1], color[mask == 2] = [255, 60, 60], [40, 220, 80]
        gt = Image.fromarray(color).resize((160, 160), Image.Resampling.NEAREST)
        overlay = Image.blend(post, gt, .4)
        if predictions is not None and identifier in predictions:
            pred = predictions[identifier]
            edges = cv2.morphologyEx((pred > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            array = np.array(overlay.resize((mask.shape[1], mask.shape[0])))
            array[edges > 0] = [255, 230, 0]
            overlay = Image.fromarray(array).resize((160, 160))
        y = i * 186
        draw.text((3, y + 2), f"{category}: {identifier} | PRE / POST / GT / overlay (pred edge yellow)", fill="black")
        for column, image in enumerate((pre, post, gt, overlay)):
            canvas.paste(image, (column * 160, y + 24))
    canvas.save(output, quality=72, optimize=True)
    return {"file": Path(output).name, "rows": chosen, "bytes": Path(output).stat().st_size}


def baseline_analysis(rows, masks, val_ids, checkpoint, config, output):
    """Existing weights, CPU inference only. No optimizer and no remote fetching."""
    import torch
    from shapely import Polygon, union_all
    from terradelta.inference.predictor import Predictor
    from terradelta.metrics import evaluate_predictions
    from terradelta.models.checkpoint import load_checkpoint
    from terradelta.models.factory import build_model
    from terradelta.postprocess import predictions_to_polygons
    from terradelta.training.validation import ground_truth_row, prediction_row

    model = build_model({"encoder_weights": None})
    metadata = load_checkpoint(checkpoint, model)
    predictor = Predictor(model, config, "cpu")
    predictions, truth, details, decoded = [], [], [], {}
    dataset = ChangeDataset(rows)
    for index, row in enumerate(rows):
        if row["id"] not in masks or np.any(masks[row["id"]] == -100):
            continue
        sample = dataset[index]
        with torch.inference_mode():
            probabilities = predictor.predict_batch(sample["image"].unsqueeze(0))[0]
        prediction = prediction_row(row["id"], probabilities, config)
        gt = ground_truth_row(sample)
        predictions.append(prediction)
        truth.append(gt)
        decoded[row["id"]] = probabilities.argmax(axis=0).astype(np.uint8)
        polygons = predictions_to_polygons(probabilities, config)
        detail = {"id": row["id"], "split": "val" if row["id"] in val_ids else "train_diagnostic",
                  "classes": {}}
        for class_index, name in enumerate(MASK_CLASSES, 1):
            pred_geometry = union_all([Polygon(ring) for ring in polygons[name]])
            gt_geometry = union_all([Polygon(ring) for ring in json.loads(gt[name])]) if gt[name] else union_all([])
            gt_positive, pred_positive = gt_geometry.area > 0, pred_geometry.area >= 20
            union_area = pred_geometry.union(gt_geometry).area
            iou = pred_geometry.intersection(gt_geometry).area / union_area if union_area else 1.0
            detail["classes"][name] = {"gt_positive": gt_positive, "pred_positive": pred_positive,
                                     "fp": not gt_positive and pred_positive, "fn": gt_positive and not pred_positive,
                                     "prediction_area": float(pred_geometry.area), "gt_area": float(gt_geometry.area),
                                     "polygon_count": len(polygons[name]), "iou": iou,
                                     "mean_confidence": float(probabilities[class_index].mean()),
                                     "p95_confidence": float(np.percentile(probabilities[class_index], 95))}
        details.append(detail)

    def summarize(ids):
        selected = [d for d in details if d["id"] in ids]
        score = evaluate_predictions([p for p in predictions if p["id"] in ids], [g for g in truth if g["id"] in ids]) if selected else None
        no_change = [d for d in selected if not any(c["gt_positive"] for c in d["classes"].values())]
        return {"metric_label": "approximate local metric", "mock_only": all(is_mock(r) for r in rows if r["id"] in ids),
                "total_validation_samples": len(selected), "metric": score,
                "no_change_samples": len(no_change),
                "no_change_fp_rate": sum(any(c["pred_positive"] for c in d["classes"].values()) for d in no_change) / len(no_change) if no_change else None,
                "average_prediction_area": {name: float(np.mean([d["classes"][name]["prediction_area"] for d in selected])) if selected else None for name in MASK_CLASSES},
                "average_polygon_count": {name: float(np.mean([d["classes"][name]["polygon_count"] for d in selected])) if selected else None for name in MASK_CLASSES},
                "note": "Mock scores cannot predict leaderboard performance or establish training readiness."}

    validation = summarize(val_ids)
    validation.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_metadata=metadata,
                      optimizer_constructed=False, optimizer_steps_executed=0, device="cpu", gradients_enabled=False)
    atomic_json(output / "baseline_validation.json", validation)
    atomic_json(output / "baseline_diagnostic.json", {**summarize(set(masks)), "samples": details,
                "scope": "all usable samples; training-partition predictions are diagnostics, not held-out validation"})
    write_csv(output / "baseline_predictions.csv", predictions, ["id", *MASK_CLASSES])
    errors = {}
    selections = []
    for name in MASK_CLASSES:
        for kind in ("fp", "fn"):
            ranked = sorted([d for d in details if d["classes"][name][kind]],
                            key=lambda d: d["classes"][name]["prediction_area" if kind == "fp" else "gt_area"], reverse=True)
            key = f"{name}_{kind}"
            errors[key] = [d["id"] for d in ranked[:3]]
            if ranked:
                selections.append((key, ranked[0]["id"]))
        boundary = sorted([d for d in details if d["classes"][name]["gt_positive"] and d["classes"][name]["pred_positive"] and d["classes"][name]["iou"] < .8],
                          key=lambda d: d["classes"][name]["iou"])
        errors[name + "_boundary_failure"] = [d["id"] for d in boundary[:3]]
        if boundary:
            selections.append((name + "_boundary", boundary[0]["id"]))
    errors["class_confusion"] = [d["id"] for d in details if any(d["classes"][a]["fn"] and d["classes"][b]["fp"] for a, b in [MASK_CLASSES, MASK_CLASSES[::-1]])]
    for cause in ("shadow", "seasonal", "registration"):
        errors[cause + "_false_positives"] = {"status": "unassessable", "reason": "No reviewed real-world cause annotations; do not infer cause from mock predictions."}
    atomic_json(output / "error_analysis.json", errors)
    return validation, montage(rows, masks, output / "baseline_errors.jpg", selections, decoded)


def audit_dataset(manifest, output, *, train_manifest=None, val_manifest=None, checkpoint=None, config=None):
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError("Audit manifest is empty")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Select a new audit directory; previous evidence is preserved")
    output.mkdir(parents=True)
    if bool(train_manifest) != bool(val_manifest):
        raise ValueError("Supply both train and validation manifests")
    split_errors = []
    if train_manifest:
        parts = {"train": read_manifest(train_manifest), "val": read_manifest(val_manifest)}
    else:
        try:
            parts = split_manifest(rows, strategy="region", seed=0)
        except ValueError as error:
            parts = {"train": [], "val": []}
            split_errors.append(str(error))
    partitions = {}
    indexed = {r["id"]: r for r in rows}
    for split, subset in parts.items():
        for row in subset:
            if row["id"] in partitions:
                split_errors.append("Duplicate partition ID: " + row["id"])
            if row["id"] not in indexed or any(row.get(k) != indexed[row["id"]].get(k) for k in PATH_FIELDS):
                split_errors.append("Partition differs from inventory: " + row["id"])
            partitions[row["id"]] = split
    if set(partitions) != set(indexed):
        split_errors.append("Split does not cover dataset exactly")
    if not all(parts.values()):
        split_errors.append("Both partitions must be nonempty")

    masks, issues, shifts, synthesis, hashes, pair_hashes = {}, [], [], [], {}, {}
    audited = []
    for row in rows:
        audited_row = provenance(row)
        raw, mask, sample_issues, estimate, synthetic = inspect_sample(row)
        usable = not any(severity == "error" for severity, _ in sample_issues)
        if mask is not None and usable:
            masks[row["id"]] = mask
        if not usable:
            audited_row.update(audit_status="unverified", audit_training_eligible=False,
                               audit_reason=audited_row["audit_reason"] + "; image/mask integrity error")
        audited.append(audited_row)
        issues.extend({"id": row["id"], "severity": severity, "reason": reason} for severity, reason in sample_issues)
        if estimate:
            shifts.append({"id": row["id"], **estimate})
        if synthetic:
            synthesis.append({"id": row["id"], **synthetic})
        if "pre" in raw and "post" in raw:
            keys = tuple(image_fingerprint(raw[key]) for key in ("pre", "post"))
            if keys in pair_hashes:
                issues.append({"id": row["id"], "severity": "warning", "reason": "duplicate pair: " + pair_hashes[keys]})
            pair_hashes[keys] = row["id"]
            for key, digest in zip(("pre", "post"), keys):
                hashes.setdefault(digest, []).append((row["id"], key))

    duplicates, content_leaks = [], []
    for references in hashes.values():
        if len(references) > 1:
            duplicates.append({"references": references})
            for a, _ in references:
                for b, _ in references:
                    if partitions.get(a) == "train" and partitions.get(b) == "val":
                        content_leaks.append({"train_id": a, "val_id": b, "reason": "identical decoded image content"})
    leak = leakage(rows, partitions)
    leak["spatial_candidates"].extend(content_leaks)
    if leak["spatial_candidates"]:
        leak["status"] = "leakage_detected"
    leak["split_errors"] = split_errors
    leak["train_samples"], leak["validation_samples"] = len(parts["train"]), len(parts["val"])
    classes = distribution(masks)
    classes.update(total_samples=len(rows), invalid_samples=len(rows) - len(masks), mock_samples=sum(is_mock(r) for r in rows))
    reliable = [s["magnitude"] for s in shifts if s["reliable"]]
    alignment = {"method": "phase correlation (analysis only)", "images_modified": False,
                 "estimated_samples": len(shifts), "reliable_samples": len(reliable),
                 "unavailable_or_unreliable": len(rows) - len(reliable),
                 "magnitude_pixels": {name: float(value) for name, value in zip(("median", "mean", "p90", "p95", "p99", "max"),
                    [np.median(reliable), np.mean(reliable), *np.percentile(reliable, [90, 95, 99]), max(reliable)])} if reliable else None,
                 "high_shift_ids": [s["id"] for s in shifts if s["reliable"] and s["magnitude"] > 4]}
    write_manifest(audited, output / "audited_manifest.csv")
    write_csv(output / "sample_provenance.csv", audited, ["id", "source", "pre_source", "post_source", "mask_source", "region_id", "bounds", "crs",
              *RASTER_FIELDS, "year_pre", "year_post", "resolution_m", "synthetic", "mock", "license_source", "license_status", "audit_status", "audit_training_eligible", "audit_reason"])
    write_csv(output / "suspicious_samples.csv", issues, ["id", "severity", "reason"])
    write_csv(output / "spatial_leakage.csv", leak["spatial_candidates"], ["train_id", "val_id", "reason"])
    write_csv(output / "temporal_leakage.csv", leak["temporal_candidates"], ["train_id", "val_id", "reason", "train_interval", "val_interval"])
    write_csv(output / "alignment.csv", shifts, ["id", "dx", "dy", "magnitude", "response", "reliable"])
    write_csv(output / "sample_distribution.csv", classes["samples"], ["id", "category", "valid_pixels", "ignored_pixels",
              "new_building_pixels", "tree_removal_pixels", "new_building_components", "tree_removal_components"])
    for filename, payload in (("class_distribution.json", classes), ("split_audit.json", leak), ("alignment.json", alignment),
                              ("synthetic_audit.json", synthesis), ("duplicate_images.json", duplicates)):
        atomic_json(output / filename, payload)
    selections = []
    for category in ("background_only", "new_building_only", "tree_removal_only", "both_classes"):
        match = next((d["id"] for d in classes["samples"] if d["category"] == category), None)
        if match:
            selections.append((category, match))
    for category, ids in (("synthetic", [r["id"] for r in audited if r["synthetic"]]),
                          ("hard_negative", [r["id"] for r in rows if r.get("negative_type")]),
                          ("suspicious", [r["id"] for r in issues if r["severity"] != "info"]),
                          ("high_shift", alignment["high_shift_ids"])):
        match = next((identifier for identifier in ids if identifier in masks), None)
        if match:
            selections.append((category, match))
    previews = [p for p in [montage(rows, masks, output / "preview.jpg", selections)] if p]
    validation = None
    if checkpoint and not split_errors:
        validation, preview = baseline_analysis(rows, masks, {r["id"] for r in parts["val"]}, checkpoint, config, output)
        if preview:
            previews.append(preview)
    blockers = []
    if not classes["background_only"]:
        blockers.append("No reviewed no-change samples for false-positive control")
    for label in MASK_CLASSES:
        if not classes["pixel_count"][label]:
            blockers.append("No reviewed " + label + " masks; target class is missing")
    if not rows or all(is_mock(r) for r in rows):
        blockers.append("No real training dataset; available samples are mock/demo only")
    if any(not r["audit_training_eligible"] for r in audited):
        blockers.append("Unverified provenance/license/label review; excluded from training readiness")
    if len(masks) != len(rows):
        blockers.append("Image/mask integrity errors")
    if split_errors or leak["errors"] or leak["spatial_candidates"] or leak["status"] == "unverified":
        blockers.append("Geographic/temporal isolation is not established or leakage exists")
    if any(np.any(mask == -100) for identifier, mask in masks.items() if partitions.get(identifier) == "val"):
        blockers.append("Validation contains partially reviewed masks")
    suspicious = {r["id"] for r in issues if r["severity"] != "info"}
    payload_paths = {Path(row[key]) for row in rows for key in PATH_FIELDS if row.get(key) and row[key].lower() != "absent" and Path(row[key]).is_file()}
    report = {"readiness": "NOT READY" if blockers else "READY WITH WARNINGS" if suspicious else "READY",
              "blockers": blockers, "manifest": str(Path(manifest).resolve()), "total_samples": len(rows),
              "real_samples": sum(not is_mock(r) for r in rows), "mock_samples": sum(is_mock(r) for r in rows),
              "sources": sorted({r.get("source", "unverified") for r in rows}),
              "dataset_image_mask_bytes": sum(path.stat().st_size for path in payload_paths),
              "manifest_bytes": Path(manifest).stat().st_size,
              "training_eligible_samples": sum(r["audit_training_eligible"] for r in audited),
              "suspicious_samples": len(suspicious), "informational_samples": len({r["id"] for r in issues if r["severity"] == "info"}),
              "baseline_validation": validation, "previews": previews,
              "network": {"dataset_download_bytes": 0, "external_metadata_download_bytes": 0, "large_transfer": False,
                          "note": "Audit reads existing local files only. Git delivery traffic is reported separately."},
              "training_executed": False, "optimizer_constructed": False, "optimizer_steps": 0,
              "aws_executed": False, "alignment_applied": False}
    atomic_json(output / "report.json", report)
    return report
