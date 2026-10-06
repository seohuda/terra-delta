#!/usr/bin/env python3
"""Ensemble evaluation: V2.3.2 Baseline (E0) vs Satlas V3 (E1) vs Hybrid Ensemble (E2)."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terradelta.data.dataset_v2 import read_v2_manifest
from terradelta.inference.v2 import V2Predictor
from terradelta.metrics.evaluation import _polygon_union, evaluate_predictions
from terradelta.models.satlas_v3 import SatlasV3Model
from terradelta.postprocess.polygons import mask_to_polygons, mask_to_polygons_reference, serialize_polygons

CLASSES = ("new_building", "tree_removal")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
def run_v232_inference(
    v2_model_path: str | Path,
    config_path: str | Path,
    samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    predictor = V2Predictor(v2_model_path, config, device)

    all_seg, all_pres = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors = []
        for s in batch:
            pre_norm = (s["pre"].astype(np.float32) / 255.0 - MEAN) / STD
            post_norm = (s["post"].astype(np.float32) / 255.0 - MEAN) / STD
            pre_t = torch.from_numpy(pre_norm.transpose(2, 0, 1)).float()
            post_t = torch.from_numpy(post_norm.transpose(2, 0, 1)).float()
            tensors.append(torch.cat([pre_t, post_t], dim=0))
        images = torch.stack(tensors).to(device)
        seg, pres = predictor.probabilities(images)
        seg_arr = seg.cpu().numpy() if isinstance(seg, torch.Tensor) else np.asarray(seg)
        pres_arr = pres.cpu().numpy() if isinstance(pres, torch.Tensor) else np.asarray(pres)
        all_seg.append(seg_arr)
        all_pres.append(pres_arr)

    return np.concatenate(all_seg, axis=0), np.concatenate(all_pres, axis=0)


@torch.inference_mode()
def run_satlas_inference(
    satlas_checkpoint: str | Path,
    samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(satlas_checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = SatlasV3Model(weights_path=None, fpn_channels=128, num_classes=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_seg, all_pres = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors = []
        for s in batch:
            pre_t = torch.from_numpy(s["pre"].transpose(2, 0, 1)).float() / 255.0
            post_t = torch.from_numpy(s["post"].transpose(2, 0, 1)).float() / 255.0
            tensors.append(torch.cat([pre_t, post_t], dim=0))
        images = torch.stack(tensors).to(device)
        seg_logits, pres_logits = model(images)
        all_seg.append(torch.sigmoid(seg_logits).cpu().numpy())
        all_pres.append(torch.sigmoid(pres_logits).cpu().numpy())

    return np.concatenate(all_seg, axis=0), np.concatenate(all_pres, axis=0)


def evaluate_probs(
    samples: list[dict],
    seg_probs: np.ndarray,
    pres_probs: np.ndarray,
    pixel_thresholds: dict[str, float],
    presence_thresholds: dict[str, float],
) -> dict[str, Any]:
    min_area = {"new_building": 30.0, "tree_removal": 20.0}
    min_pos_area = {"new_building": 20.0, "tree_removal": 20.0}

    predictions = []
    truth = []
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
                simplify_px=0.5,
                backend="reference",
            )
            row[c_name] = serialize_polygons(polygons)
        predictions.append(row)
        truth.append({
            "id": s["id"],
            "new_building": mask_to_polygons_reference(s["target_bldg"], 0, 0, 0),
            "tree_removal": mask_to_polygons_reference(s["target_tree"], 0, 0, 0),
        })

    raw_metrics = evaluate_predictions(predictions, truth)
    predicted_map = {r["id"]: r for r in predictions}
    negatives = [r["id"] for r in truth if all(_polygon_union(r[c], "gt").is_empty for c in CLASSES)]
    no_change_fp = sum(
        1 for nid in negatives
        if any(_polygon_union(predicted_map[nid][c], "pred").area >= 20.0 for c in CLASSES)
    )
    bldg_tp = sum(
        1 for r in truth
        if not _polygon_union(r["new_building"], "gt").is_empty
        and not _polygon_union(predicted_map[r["id"]]["new_building"], "pred").is_empty
    )
    bldg_pos = sum(1 for r in truth if not _polygon_union(r["new_building"], "gt").is_empty)
    tree_tp = sum(
        1 for r in truth
        if not _polygon_union(r["tree_removal"], "gt").is_empty
        and not _polygon_union(predicted_map[r["id"]]["tree_removal"], "pred").is_empty
    )
    tree_pos = sum(1 for r in truth if not _polygon_union(r["tree_removal"], "gt").is_empty)

    return {
        "score": raw_metrics.get("score", 0.0),
        "no_change_fp": no_change_fp,
        "no_change_count": len(negatives),
        "bldg_tp": bldg_tp,
        "bldg_positives": bldg_pos,
        "tree_tp": tree_tp,
        "tree_positives": tree_pos,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble: V2.3.2 vs Satlas V3 vs Hybrid")
    parser.add_argument("--satlas-ckpt", type=str, required=True)
    parser.add_argument("--v2-model", type=str, default="/data/terradelta/releases/v2.2-main-01/model.pt")
    parser.add_argument("--v2-config", type=str, default="/data/terradelta/releases/v2.2-main-01/config.yaml")
    parser.add_argument("--legacy-manifest", type=str, default="/data/terradelta/processed/micro-pilot-v2/val.csv")
    parser.add_argument("--stress-manifest", type=str, default="/data/terradelta/v2/manifests/stress-val-v2.csv")
    parser.add_argument("--satlas-weight", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading datasets on {device}...")
    legacy_samples = load_dataset_samples(args.legacy_manifest)
    stress_samples = load_dataset_samples(args.stress_manifest)

    # Thresholds
    pixel_thresh = {"new_building": 0.35, "tree_removal": 0.70}
    pres_thresh = {"new_building": 0.30, "tree_removal": 0.30}

    # 1. Run V2.3.2
    print("Running V2.3.2 inference...")
    v2_leg_seg, v2_leg_pres = run_v232_inference(args.v2_model, args.v2_config, legacy_samples, device)
    v2_str_seg, v2_str_pres = run_v232_inference(args.v2_model, args.v2_config, stress_samples, device)
    e0_leg = evaluate_probs(legacy_samples, v2_leg_seg, v2_leg_pres, pixel_thresh, pres_thresh)
    e0_str = evaluate_probs(stress_samples, v2_str_seg, v2_str_pres, pixel_thresh, pres_thresh)

    # 2. Run Satlas V3
    print("Running Satlas V3 inference...")
    sat_leg_seg, sat_leg_pres = run_satlas_inference(args.satlas_ckpt, legacy_samples, device)
    sat_str_seg, sat_str_pres = run_satlas_inference(args.satlas_ckpt, stress_samples, device)
    e1_leg = evaluate_probs(legacy_samples, sat_leg_seg, sat_leg_pres, pixel_thresh, pres_thresh)
    e1_str = evaluate_probs(stress_samples, sat_str_seg, sat_str_pres, pixel_thresh, pres_thresh)

    # 3. Ensemble (Blended probabilities)
    w = args.satlas_weight
    print(f"Blending ensemble with Satlas weight {w}...")
    ens_leg_seg = w * sat_leg_seg + (1 - w) * v2_leg_seg
    ens_leg_pres = w * sat_leg_pres + (1 - w) * v2_leg_pres
    ens_str_seg = w * sat_str_seg + (1 - w) * v2_str_seg
    ens_str_pres = w * sat_str_pres + (1 - w) * v2_str_pres
    e2_leg = evaluate_probs(legacy_samples, ens_leg_seg, ens_leg_pres, pixel_thresh, pres_thresh)
    e2_str = evaluate_probs(stress_samples, ens_str_seg, ens_str_pres, pixel_thresh, pres_thresh)

    report = {
        "E0_V232": {"legacy": e0_leg, "stress": e0_str},
        "E1_Satlas": {"legacy": e1_leg, "stress": e1_str},
        "E2_Ensemble": {"legacy": e2_leg, "stress": e2_str, "satlas_weight": w},
    }

    print("\n=================== ENSEMBLE COMPARISON ===================")
    print(f"E0 (V2.3.2):    Real: {e0_leg['score']:.4f} (Bldg {e0_leg['bldg_tp']}/3, Tree {e0_leg['tree_tp']}/2, FP {e0_leg['no_change_fp']}/17) | Stress: {e0_str['score']:.4f} (Bldg {e0_str['bldg_tp']}, Tree {e0_str['tree_tp']}, FP {e0_str['no_change_fp']})")
    print(f"E1 (Satlas V3): Real: {e1_leg['score']:.4f} (Bldg {e1_leg['bldg_tp']}/3, Tree {e1_leg['tree_tp']}/2, FP {e1_leg['no_change_fp']}/17) | Stress: {e1_str['score']:.4f} (Bldg {e1_str['bldg_tp']}, Tree {e1_str['tree_tp']}, FP {e1_str['no_change_fp']})")
    print(f"E2 (Ensemble):  Real: {e2_leg['score']:.4f} (Bldg {e2_leg['bldg_tp']}/3, Tree {e2_leg['tree_tp']}/2, FP {e2_leg['no_change_fp']}/17) | Stress: {e2_str['score']:.4f} (Bldg {e2_str['bldg_tp']}, Tree {e2_str['tree_tp']}, FP {e2_str['no_change_fp']})")
    print("===========================================================\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved ensemble report to {out_p}")


if __name__ == "__main__":
    main()
