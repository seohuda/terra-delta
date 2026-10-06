"""Training object-level evidence classifier for candidate change components.

Trains class-independent lightweight regularized logistic regression models
(new_building and tree_removal) to classify candidate TP vs FP.

Training labels:
- Real reviewed full negatives: all candidate components labeled FP (y=0)
- Real partial positives: candidates overlapping valid positive mask labeled TP (y=1);
  components in unreviewed areas labeled UNKNOWN (-1, excluded)
- Synthetic samples: overlap/IoU rules for conservative TP/FP, ambiguous excluded.
Never trains on legacy 22 or stress validation pairs.
Grouped cross-validation by scene/source to prevent spatial/source leakage.
Exports models to pure NumPy JSON format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.inference.evidence_classifier import EvidenceClassifier, LinearComponentClassifier
from terradelta.inference.evidence_features import (
    ABLATION_SCHEMAS,
    extract_candidate_evidence_record,
    extract_deep_cva_maps,
    get_component_pixels,
)
from terradelta.inference.stability_v23 import (
    component_features as stability_component_features,
    tta_probabilities,
)
from terradelta.inference.v2 import CLASSES, V2Predictor, class_options
from terradelta.inference.v231_predictor import (
    compute_global_metrics,
    compute_reverse_features,
    reverse_pair_tensor,
    tensor_to_rgb,
)
from terradelta.inference.verifier import verify_frozen_checkpoint
from terradelta.postprocess.polygons import reference_components
from terradelta.utils.io import atomic_json


def assign_candidate_label(
    part_mask: tuple[np.ndarray, np.ndarray],
    poly_area: float,
    gt_target: np.ndarray,
    review_mask: np.ndarray | None,
    category: str,
) -> int:
    """Assign conservative training label for a candidate component.

    Returns:
       1: True Positive (TP)
       0: False Positive (FP)
      -1: Ambiguous / Unknown (EXCLUDE from training)
    """
    y, x = part_mask
    if len(y) == 0:
        return -1

    # Case A: Real confirmed negative pair
    if category in ("real_negative", "negative"):
        return 0

    gt_pixels_in_part = gt_target[y, x] > 0
    overlap_count = int(gt_pixels_in_part.sum())
    overlap_fraction = overlap_count / max(len(y), 1)

    # Case B: Partial review mask present
    if review_mask is not None and not review_mask.all():
        reviewed_pixels = review_mask[y, x] > 0
        reviewed_fraction = float(reviewed_pixels.mean())
        if overlap_fraction >= 0.40:
            return 1  # Overlaps confirmed real change
        if reviewed_fraction >= 0.95 and overlap_fraction < 0.05:
            return 0  # Confirmed within reviewed negative area
        return -1  # Falls in unreviewed / ambiguous area -> UNKNOWN

    # Case C: Fully labeled synthetic / real positive
    total_gt = int((gt_target > 0).sum())
    if total_gt == 0:
        return 0

    intersection = overlap_count
    union = len(y) + total_gt - intersection
    iou = intersection / max(union, 1)

    if overlap_fraction >= 0.40 or iou >= 0.25:
        return 1
    elif overlap_fraction < 0.05:
        return 0
    else:
        return -1  # Boundary / partial overlap ambiguity -> EXCLUDE


@torch.inference_mode()
def extract_dataset_candidates(
    data: IndependentChangeDataset,
    predictor: V2Predictor,
    config: Mapping[str, Any],
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Extract candidate components with multi-group evidence and training labels."""
    candidates = []
    started = time.monotonic()
    total = len(data)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        samples = [data[i] for i in range(start, end)]
        image_batch = torch.stack([s["image"] for s in samples])

        pixels_batch, presence_batch = predictor.probabilities(image_batch)
        v22_rows = predictor.output_rows(
            [s["id"] for s in samples], image_batch, pixels_batch, presence_batch
        )

        has_any_candidate = any(row.get(c) for row in v22_rows for c in CLASSES)
        if not has_any_candidate:
            if (start + batch_size) % 80 == 0 or end == total:
                print(
                    json.dumps({
                        "event": "extract_progress",
                        "done": end,
                        "total": total,
                        "candidates": len(candidates),
                        "elapsed_sec": round(time.monotonic() - started, 1),
                    }),
                    flush=True,
                )
            continue

        tta_pixels, tta_heads = tta_probabilities(predictor, image_batch, (pixels_batch, presence_batch))
        rev_images = reverse_pair_tensor(image_batch)
        rev_pixels, rev_heads = predictor.probabilities(rev_images)
        cva_maps, cos_maps = extract_deep_cva_maps(
            predictor.model, image_batch[:, :3], image_batch[:, 3:], predictor.device
        )
        pre_rgbs, post_rgbs = tensor_to_rgb(image_batch)

        for b_idx, sample in enumerate(samples):
            pair_id = sample["id"]
            row_meta = data.rows[start + b_idx]
            category = row_meta.get("category", "")
            source_id = row_meta.get("source_id", row_meta.get("region", pair_id))
            review_mask = sample.get("review_mask")
            if review_mask is not None:
                review_mask = review_mask.numpy()

            global_metrics = compute_global_metrics(pre_rgbs[b_idx], post_rgbs[b_idx])
            stab_features = stability_component_features(
                tta_pixels[:, b_idx], tta_heads[:, b_idx], config
            )

            for channel, class_name in enumerate(CLASSES):
                options = class_options(config, class_name)
                comp_list = stab_features[class_name]
                if not comp_list or not v22_rows[b_idx].get(class_name):
                    continue

                originals = [
                    p
                    for p in reference_components(pixels_batch[b_idx, channel] >= options["pixel_threshold"])
                    if p.area >= options["min_area"]
                ]
                if sum(p.area for p in originals) < options["min_pos_area"]:
                    continue

                rev_prob = rev_pixels[b_idx, channel]
                rev_pres = float(rev_heads[b_idx, channel])
                rev_parts = [
                    p
                    for p in reference_components(rev_prob >= options["pixel_threshold"])
                    if p.area >= options["min_area"]
                ]

                gt_target = sample["target"][channel].numpy()

                for k, (part, stab_rec) in enumerate(zip(originals, comp_list)):
                    y, x = get_component_pixels(part)
                    label = assign_candidate_label(
                        (y, x), part.area, gt_target, review_mask, category
                    )

                    rev_rec = compute_reverse_features(
                        y, x, part, rev_prob, rev_pres, rev_parts, options["pixel_threshold"]
                    )

                    evidence = extract_candidate_evidence_record(
                        poly=part,
                        component_index=k,
                        y=y,
                        x=x,
                        all_polys=originals,
                        identity_prob=pixels_batch[b_idx, channel],
                        identity_presence=float(presence_batch[b_idx, channel]),
                        verifier_score=0.0,
                        stability_rec=stab_rec,
                        reverse_rec=rev_rec,
                        cva_map=cva_maps[b_idx],
                        cosine_map=cos_maps[b_idx],
                        pre_rgb=pre_rgbs[b_idx],
                        post_rgb=post_rgbs[b_idx],
                        global_metrics=global_metrics,
                        class_name=class_name,
                    )

                    candidates.append({
                        "pair_id": pair_id,
                        "source_id": source_id,
                        "class_name": class_name,
                        "label": label,
                        "features": evidence["features"],
                    })

        if (start + batch_size) % 80 == 0 or end == total:
            print(
                json.dumps({
                    "event": "extract_progress",
                    "done": end,
                    "total": total,
                    "candidates": len(candidates),
                    "elapsed_sec": round(time.monotonic() - started, 1),
                }),
                flush=True,
            )

    return candidates


def fit_logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    l2_penalty: float = 1.0,
    max_iter: int = 50,
) -> tuple[np.ndarray, float]:
    """Pure NumPy IRLS regularized logistic regression solver."""
    n, d = X.shape
    if sample_weight is None:
        # Balanced sample weights by default
        n_pos = max(1, int(y.sum()))
        n_neg = max(1, int((y == 0).sum()))
        sample_weight = np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))
    design = np.column_stack([X, np.ones(n)])
    beta = np.zeros(d + 1)
    penalty = np.diag([l2_penalty] * d + [0.0])
    for _ in range(max_iter):
        logits = np.clip(design @ beta, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = design.T @ (sample_weight * (prob - y)) + penalty @ beta
        w_diag = sample_weight * np.maximum(prob * (1.0 - prob), 1e-6)
        hessian = design.T @ (w_diag[:, None] * design) + penalty + np.eye(d + 1) * 1e-6
        step = np.linalg.solve(hessian, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta[:-1], float(beta[-1])


def group_kfold_splits(groups: np.ndarray, n_splits: int = 5):
    """Deterministic grouped k-fold generator."""
    unique_groups = np.sort(np.unique(groups))
    folds = np.array_split(unique_groups, n_splits)
    for fold in folds:
        val_mask = np.isin(groups, fold)
        yield np.where(~val_mask)[0], np.where(val_mask)[0]


def train_single_class_classifier(
    candidates: list[dict[str, Any]],
    class_name: str,
    feature_schema: Sequence[str],
    c_reg: float = 0.5,
) -> tuple[LinearComponentClassifier, dict[str, Any]]:
    """Train regularized logistic regression with GroupKFold cross-validation."""
    eligible = [c for c in candidates if c["class_name"] == class_name and c["label"] in (0, 1)]
    if not eligible:
        raise ValueError(f"No labeled candidates found for {class_name}")

    X = np.empty((len(eligible), len(feature_schema)), dtype=np.float64)
    y = np.array([c["label"] for c in eligible], dtype=int)
    groups = np.array([c["source_id"] for c in eligible])

    for i, c in enumerate(eligible):
        feat_map = c["features"]
        for j, fname in enumerate(feature_schema):
            val = feat_map.get(fname, 0.0)
            X[i, j] = val if np.isfinite(val) else 0.0

    # Grouped cross-validation
    n_groups = len(set(groups))
    n_splits = min(5, max(2, n_groups))

    oof_preds = np.zeros(len(y), dtype=float)
    l2_pen = 1.0 / max(c_reg, 1e-3)
    for train_idx, val_idx in group_kfold_splits(groups, n_splits=n_splits):
        mean_tr = np.mean(X[train_idx], axis=0)
        std_tr = np.std(X[train_idx], axis=0)
        scale_tr = np.where(std_tr > 1e-6, std_tr, 1.0)

        X_tr_norm = (X[train_idx] - mean_tr) / scale_tr
        X_val_norm = (X[val_idx] - mean_tr) / scale_tr

        w, b = fit_logistic_irls(X_tr_norm, y[train_idx], l2_penalty=l2_pen)
        logits_val = np.clip(X_val_norm @ w + b, -30, 30)
        oof_preds[val_idx] = 1.0 / (1.0 + np.exp(-logits_val))

    # Evaluate OOF across threshold grid to find broad stable plateau
    threshold_metrics = []
    for th in np.arange(0.20, 0.81, 0.05):
        pred_pos = oof_preds >= th
        tp = int(((pred_pos == 1) & (y == 1)).sum())
        fp = int(((pred_pos == 1) & (y == 0)).sum())
        fn = int(((pred_pos == 0) & (y == 1)).sum())
        tn = int(((pred_pos == 0) & (y == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        fp_reduction = (len(y) - y.sum() - fp) / max(len(y) - y.sum(), 1)
        threshold_metrics.append({
            "threshold": round(float(th), 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fp_reduction": round(fp_reduction, 4),
        })

    # Pick threshold with high recall and maximum FP reduction
    # Conservative threshold bounds prevent over-filtering out-of-distribution real changes
    if class_name == "tree_removal":
        candidates_th = [m for m in threshold_metrics if m["recall"] >= 0.94 and m["threshold"] <= 0.20]
        if candidates_th:
            best_metric = max(candidates_th, key=lambda m: (m["fp_reduction"], m["f1"]))
        else:
            best_metric = next(m for m in threshold_metrics if m["threshold"] == 0.20)
    else:
        candidates_th = [m for m in threshold_metrics if m["recall"] >= 0.88 and m["threshold"] <= 0.55]
        if candidates_th:
            best_metric = max(candidates_th, key=lambda m: (m["fp_reduction"], m["f1"]))
        else:
            best_metric = max(threshold_metrics, key=lambda m: m["f1"])
    selected_threshold = best_metric["threshold"]

    # Final fit on all data
    mean_full = np.mean(X, axis=0)
    std_full = np.std(X, axis=0)
    scale_full = np.where(std_full > 1e-6, std_full, 1.0)
    X_full_norm = (X - mean_full) / scale_full

    final_w, final_b = fit_logistic_irls(X_full_norm, y, l2_penalty=l2_pen)

    model = LinearComponentClassifier(
        feature_names=feature_schema,
        mean=mean_full.tolist(),
        scale=scale_full.tolist(),
        weights=final_w.tolist(),
        intercept=final_b,
        threshold=selected_threshold,
    )

    report = {
        "class_name": class_name,
        "n_samples": len(y),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "n_groups": int(n_groups),
        "selected_threshold": selected_threshold,
        "oof_metrics_at_selected": best_metric,
        "threshold_grid": threshold_metrics,
        "top_positive_weights": [
            (feature_schema[idx], round(float(final_w[idx]), 4))
            for idx in np.argsort(-final_w)[:8]
        ],
        "top_negative_weights": [
            (feature_schema[idx], round(float(final_w[idx]), 4))
            for idx in np.argsort(final_w)[:8]
        ],
    }

    return model, report


TREE_CONFOUNDER_EXCLUDES = {
    "num_same_class_components",
    "total_class_area",
    "global_rgb_shift",
    "global_edge_diff",
    "global_brightness_shift",
    "global_contrast_shift",
}


def main(args):
    verify_frozen_checkpoint(args.checkpoint)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # Strict anti-leakage: exclude any sample present in legacy or stress validation
    train_ds = IndependentChangeDataset(args.train)
    legacy_ds = IndependentChangeDataset(args.legacy)
    stress_ds = IndependentChangeDataset(args.stress)
    legacy_ids = set(r["id"] for r in legacy_ds.rows)
    stress_ids = set(r["id"] for r in stress_ds.rows)
    forbidden_ids = legacy_ids.union(stress_ids)

    initial_count = len(train_ds.rows)
    clean_rows = [r for r in train_ds.rows if r["id"] not in forbidden_ids]
    excluded_count = initial_count - len(clean_rows)
    print(f"Anti-leakage: excluded {excluded_count} samples present in validation sets. Clean train: {len(clean_rows)}")
    train_ds.rows = clean_rows
    assert not (set(r["id"] for r in train_ds.rows) & forbidden_ids), "Validation leakage detected!"

    predictor = V2Predictor(args.checkpoint, config, args.device)

    cache_file = out_dir / "train_candidates_cache.json"
    if cache_file.exists():
        print(f"Loading cached training candidates from {cache_file}...")
        candidates = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        print(f"Extracting candidates from {len(train_ds)} training pairs...")
        candidates = extract_dataset_candidates(train_ds, predictor, config, batch_size=8)
        atomic_json(cache_file, candidates)

    print(f"Total extracted candidates: {len(candidates)}")
    for name in CLASSES:
        n_pos = sum(1 for c in candidates if c["class_name"] == name and c["label"] == 1)
        n_neg = sum(1 for c in candidates if c["class_name"] == name and c["label"] == 0)
        n_unk = sum(1 for c in candidates if c["class_name"] == name and c["label"] == -1)
        print(f"Class {name}: {n_pos} TP, {n_neg} FP, {n_unk} UNKNOWN (excluded)")

    # Train models for each ablation schema
    models_per_ablation = {}
    reports_per_ablation = {}

    for ablation_key in ("B", "C", "D", "E"):
        print(f"\n--- Training Ablation {ablation_key} Models ---")
        ablation_models = {}
        ablation_reports = {}
        for name in CLASSES:
            if ablation_key == "E":
                schema_key = "E_building" if "building" in name else "E_tree"
            else:
                schema_key = ablation_key
            schema = ABLATION_SCHEMAS[schema_key]
            if name == "tree_removal":
                schema = [f for f in schema if f not in TREE_CONFOUNDER_EXCLUDES]
            c_reg = 0.3 if name == "tree_removal" else 0.5
            model, report = train_single_class_classifier(candidates, name, schema, c_reg=c_reg)
            ablation_models[name] = model
            ablation_reports[name] = report
            print(
                f"[{ablation_key}] {name}: th={report['selected_threshold']}, OOF recall={report['oof_metrics_at_selected']['recall']}, FP reduction={report['oof_metrics_at_selected']['fp_reduction']}"
            )

        container = EvidenceClassifier(ablation_models, enabled=True)
        container.save(out_dir / f"classifier_ablation_{ablation_key}.json")
        models_per_ablation[ablation_key] = container
        reports_per_ablation[ablation_key] = ablation_reports

    summary = {
        "status": "training_completed",
        "checkpoint": args.checkpoint,
        "n_train_samples": len(train_ds),
        "n_candidates_total": len(candidates),
        "ablations": reports_per_ablation,
    }
    atomic_json(out_dir / "training_summary.json", summary)
    print(f"\nTraining finished successfully. Artifacts saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--audit", required=False, default=None)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--stress", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    main(parser.parse_args())
