#!/usr/bin/env python3
"""Evaluate TerraDelta V3 Satlas Models on Real Legacy and Stress Validation Sets.

Computes:
- Real Legacy (22 pairs): Building recall, Tree recall, No-change FP count, Legacy score
- Stress Validation (470 pairs): Building TP, Tree TP, No-change FP count, Stress score
- Promotion Gate checks:
    Real Bldg Recall >= 3/3
    Real Tree Recall >= 2/2
    Real No-Change FP <= 8/17
    Stress Score >= 0.680
- Optional ensembling with V2.3.2 baseline predictions
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terradelta.data.dataset_v2 import read_v2_manifest
from terradelta.metrics.evaluation import _polygon_union, evaluate_predictions
from terradelta.models.satlas_v3 import SatlasV3Model
from terradelta.postprocess.polygons import mask_to_polygons, mask_to_polygons_reference, serialize_polygons


CLASSES = ("new_building", "tree_removal")


def load_dataset_samples(manifest_path: str | Path) -> list[dict[str, Any]]:
    rows = read_v2_manifest(manifest_path)
    samples = []
    for r in rows:
        pre_img = np.array(Image.open(r["pre"]).convert("RGB"), dtype=np.uint8)
        post_img = np.array(Image.open(r["post"]).convert("RGB"), dtype=np.uint8)
        h, w = pre_img.shape[:2]

        def get_mask(p):
            if not p or p.lower() == "absent" or not Path(p).is_file():
                return np.zeros((h, w), dtype=bool)
            return np.array(Image.open(p).convert("L")) > 0

        target_bldg = get_mask(r.get("new_building"))
        target_tree = get_mask(r.get("tree_removal"))

        samples.append({
            "id": r["id"],
            "pre": pre_img,
            "post": post_img,
            "target_bldg": target_bldg,
            "target_tree": target_tree,
        })
    return samples


@torch.inference_mode()
def run_model_inference(
    model: SatlasV3Model,
    samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all segmentation probs (N, 2, 256, 256) and presence probs (N, 2)."""
    model.eval()
    all_seg = []
    all_pres = []

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        tensors = []
        for s in batch_samples:
            pre_t = torch.from_numpy(s["pre"].transpose(2, 0, 1)).float() / 255.0
            post_t = torch.from_numpy(s["post"].transpose(2, 0, 1)).float() / 255.0
            tensors.append(torch.cat([pre_t, post_t], dim=0))

        images = torch.stack(tensors).to(device)
        seg_logits, pres_logits = model(images)
        seg_prob = torch.sigmoid(seg_logits).cpu().numpy()
        pres_prob = torch.sigmoid(pres_logits).cpu().numpy()

        all_seg.append(seg_prob)
        all_pres.append(pres_prob)

    return np.concatenate(all_seg, axis=0), np.concatenate(all_pres, axis=0)


def extract_predictions(
    samples: list[dict[str, Any]],
    seg_probs: np.ndarray,
    pres_probs: np.ndarray,
    pixel_thresholds: dict[str, float],
    presence_thresholds: dict[str, float],
    min_area: dict[str, float],
    min_pos_area: dict[str, float],
    simplify_px: float = 0.5,
) -> list[dict[str, str]]:
    predictions = []
    for i, s in enumerate(samples):
        row = {"id": s["id"]}
        for c_idx, c_name in enumerate(CLASSES):
            p_thresh = pixel_thresholds.get(c_name, 0.5)
            pres_thresh = presence_thresholds.get(c_name, 0.3)

            mask = seg_probs[i, c_idx] >= p_thresh
            if pres_probs[i, c_idx] < pres_thresh:
                mask[:] = False

            polygons = mask_to_polygons(
                mask,
                min_area=float(min_area.get(c_name, 30)),
                min_pos_area=float(min_pos_area.get(c_name, 20)),
                simplify_px=simplify_px,
                backend="reference",
            )
            row[c_name] = serialize_polygons(polygons)
        predictions.append(row)
    return predictions


def build_ground_truth(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    truth = []
    for s in samples:
        row = {
            "id": s["id"],
            "new_building": mask_to_polygons_reference(s["target_bldg"], 0, 0, 0),
            "tree_removal": mask_to_polygons_reference(s["target_tree"], 0, 0, 0),
        }
        truth.append(row)
    return truth


def compute_metrics(predictions: list[dict], truth: list[dict]) -> dict[str, Any]:
    raw_metrics = evaluate_predictions(predictions, truth)
    predicted_map = {r["id"]: r for r in predictions}
    truth_map = {r["id"]: r for r in truth}

    # Negatives: ground truth empty for both classes
    negatives = [
        r["id"] for r in truth
        if all(_polygon_union(r[c], "gt").is_empty for c in CLASSES)
    ]
    no_change_count = len(negatives)
    no_change_fp_count = 0
    for nid in negatives:
        pred_row = predicted_map[nid]
        if any(_polygon_union(pred_row[c], "pred").area >= 20.0 for c in CLASSES):
            no_change_fp_count += 1

    # Class-specific counts
    bldg_tp = sum(
        1 for r in truth
        if not _polygon_union(r["new_building"], "gt").is_empty
        and not _polygon_union(predicted_map[r["id"]]["new_building"], "pred").is_empty
    )
    bldg_positives = sum(1 for r in truth if not _polygon_union(r["new_building"], "gt").is_empty)

    tree_tp = sum(
        1 for r in truth
        if not _polygon_union(r["tree_removal"], "gt").is_empty
        and not _polygon_union(predicted_map[r["id"]]["tree_removal"], "pred").is_empty
    )
    tree_positives = sum(1 for r in truth if not _polygon_union(r["tree_removal"], "gt").is_empty)

    return {
        "score": raw_metrics.get("score", 0.0),
        "classes": raw_metrics.get("classes", {}),
        "no_change_count": no_change_count,
        "no_change_fp": no_change_fp_count,
        "bldg_tp": bldg_tp,
        "bldg_positives": bldg_positives,
        "tree_tp": tree_tp,
        "tree_positives": tree_positives,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    legacy_samples: list[dict],
    stress_samples: list[dict],
    device: torch.device,
    pixel_thresholds: dict[str, float],
    presence_thresholds: dict[str, float],
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = SatlasV3Model(weights_path=None, fpn_channels=128, num_classes=2)
    model.load_state_dict(state_dict)
    model.to(device)

    min_area = {"new_building": 30.0, "tree_removal": 20.0}
    min_pos_area = {"new_building": 20.0, "tree_removal": 20.0}

    # Legacy
    leg_seg, leg_pres = run_model_inference(model, legacy_samples, device)
    leg_preds = extract_predictions(legacy_samples, leg_seg, leg_pres, pixel_thresholds, presence_thresholds, min_area, min_pos_area)
    leg_truth = build_ground_truth(legacy_samples)
    leg_metrics = compute_metrics(leg_preds, leg_truth)

    # Stress
    stress_seg, stress_pres = run_model_inference(model, stress_samples, device)
    stress_preds = extract_predictions(stress_samples, stress_seg, stress_pres, pixel_thresholds, presence_thresholds, min_area, min_pos_area)
    stress_truth = build_ground_truth(stress_samples)
    stress_metrics = compute_metrics(stress_preds, stress_truth)

    # Promotion Gates
    gate_bldg_recall = leg_metrics["bldg_tp"] >= 3  # 3/3
    gate_tree_recall = leg_metrics["tree_tp"] >= 2  # 2/2
    gate_real_fp = leg_metrics["no_change_fp"] <= 8  # <= 8/17
    gate_stress_score = stress_metrics["score"] >= 0.680  # >= 0.680

    passed_all_gates = gate_bldg_recall and gate_tree_recall and gate_real_fp and gate_stress_score

    return {
        "checkpoint": str(checkpoint_path),
        "step": ckpt.get("step", "unknown"),
        "recipe": ckpt.get("recipe", "unknown"),
        "legacy": leg_metrics,
        "stress": stress_metrics,
        "gates": {
            "bldg_recall_ge_3": gate_bldg_recall,
            "tree_recall_ge_2": gate_tree_recall,
            "real_fp_le_8": gate_real_fp,
            "stress_score_ge_068": gate_stress_score,
            "passed_all": passed_all_gates,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TerraDelta V3 Satlas Checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--legacy-manifest", type=str, default="/data/terradelta/processed/micro-pilot-v2/val.csv")
    parser.add_argument("--stress-manifest", type=str, default="/data/terradelta/v2/manifests/stress-val-v2.csv")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--pixel-thresh-bldg", type=float, default=0.35)
    parser.add_argument("--pixel-thresh-tree", type=float, default=0.70)
    parser.add_argument("--pres-thresh-bldg", type=float, default=0.30)
    parser.add_argument("--pres-thresh-tree", type=float, default=0.30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading validation datasets on {device}...")
    t0 = time.time()
    legacy_samples = load_dataset_samples(args.legacy_manifest)
    stress_samples = load_dataset_samples(args.stress_manifest)
    print(f"Loaded {len(legacy_samples)} legacy samples and {len(stress_samples)} stress samples in {time.time()-t0:.1f}s")

    pixel_thresholds = {"new_building": args.pixel_thresh_bldg, "tree_removal": args.pixel_thresh_tree}
    presence_thresholds = {"new_building": args.pres_thresh_bldg, "tree_removal": args.pres_thresh_tree}

    results = evaluate_checkpoint(
        args.checkpoint,
        legacy_samples,
        stress_samples,
        device,
        pixel_thresholds,
        presence_thresholds,
    )

    print("\n================ EVALUATION RESULTS ================")
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Real Legacy Score: {results['legacy']['score']:.6f}")
    print(f"  Building Recall: {results['legacy']['bldg_tp']}/{results['legacy']['bldg_positives']}")
    print(f"  Tree Recall:     {results['legacy']['tree_tp']}/{results['legacy']['tree_positives']}")
    print(f"  No-Change FP:    {results['legacy']['no_change_fp']}/{results['legacy']['no_change_count']}")
    print(f"Stress Score:      {results['stress']['score']:.6f}")
    print(f"  Building TP:     {results['stress']['bldg_tp']}")
    print(f"  Tree TP:         {results['stress']['tree_tp']}")
    print(f"  No-Change FP:    {results['stress']['no_change_fp']}/{results['stress']['no_change_count']}")
    print(f"Promotion Gates:   {results['gates']}")
    print("====================================================\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation report to {out_p}")


if __name__ == "__main__":
    main()
