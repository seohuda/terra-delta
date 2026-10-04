#!/usr/bin/env python3
"""Short-budget TerraDelta v2 trainer with independent mask and presence heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terradelta.data.dataset_v2 import IndependentChangeDataset, V2Transform
from terradelta.metrics import evaluate_predictions
from terradelta.models.siamese_v2 import (
    TerraDeltaSiameseV2,
    initialize_encoder_from_baseline,
    save_v2_checkpoint,
)
from terradelta.postprocess.polygons import mask_to_polygons, mask_to_polygons_reference, serialize_polygons
from terradelta.training.losses_v2 import V2Loss


def collate(samples, device):
    keys = ("image", "target", "valid_mask", "presence", "presence_valid")
    return {key: torch.stack([sample[key] for sample in samples]).to(device) for key in keys}


def gt_row(sample):
    return {
        "id": sample["id"],
        "new_building": mask_to_polygons_reference(sample["target"][0].numpy() > .5, 0, 0, 0),
        "tree_removal": mask_to_polygons_reference(sample["target"][1].numpy() > .5, 0, 0, 0),
    }


@torch.inference_mode()
def evaluate(model, dataset, device, config):
    model.eval()
    options = config["evaluation"]
    predictions, truth = [], []
    names = ("new_building", "tree_removal")
    confusion = {name: dict(tp=0, fp=0, fn=0, tn=0) for name in names}
    batch_size = int(options.get("batch_size", 8))

    for start in range(0, len(dataset), batch_size):
        samples = [dataset.get(i, seed=0) for i in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([sample["image"] for sample in samples]).to(device)
        output = model(images)
        segmentation = torch.sigmoid(output["segmentation"]).cpu().numpy()
        presence = torch.sigmoid(output["presence"]).cpu().numpy()

        for offset, sample in enumerate(samples):
            row = {"id": sample["id"]}
            for class_index, name in enumerate(names):
                mask = segmentation[offset, class_index] >= float(options["pixel_thresholds"][name])
                if presence[offset, class_index] < float(options["presence_thresholds"][name]):
                    mask[:] = False
                polygons = mask_to_polygons(
                    mask,
                    min_area=float(options.get("min_area", {}).get(name, 30)),
                    min_pos_area=float(options.get("min_pos_area", {}).get(name, 20)),
                    simplify_px=float(options.get("simplify_px", .5)),
                    backend="reference",
                )
                row[name] = serialize_polygons(polygons)
                gt_positive = bool(sample["target"][class_index].any())
                pred_positive = bool(polygons)
                key = (
                    "tp" if gt_positive and pred_positive
                    else "fp" if pred_positive
                    else "fn" if gt_positive
                    else "tn"
                )
                confusion[name][key] += 1
            predictions.append(row)
            truth.append(gt_row(sample))

    result = evaluate_predictions(predictions, truth)
    result["pair_confusion"] = confusion
    return result


def pools(dataset):
    result = {key: [] for key in ("new_building", "tree_removal", "both", "negative", "real")}
    for index, row in enumerate(dataset.rows):
        category = row.get("category", "real")
        if category not in result:
            category = "real"
        result[category].append(index)
    return result


def draw_batch(groups, batch_size, rng):
    plan = [("new_building", .25), ("tree_removal", .25), ("both", .10)]
    chosen = []
    for key, fraction in plan:
        count = max(1, int(round(batch_size * fraction))) if groups.get(key) else 0
        chosen.extend(rng.choice(groups[key]) for _ in range(count))
    remainder = max(0, batch_size - len(chosen))
    negative_pool = groups.get("negative", []) + groups.get("real", [])
    if not negative_pool:
        negative_pool = sum(groups.values(), [])
    chosen.extend(rng.choice(negative_pool) for _ in range(remainder))
    rng.shuffle(chosen)
    return chosen[:batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config.get("seed", 20261004))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    augmentation = V2Transform(**config["augmentation"])
    train = IndependentChangeDataset(config["data"]["train_manifest"], transform=augmentation)
    legacy = IndependentChangeDataset(config["data"]["legacy_val_manifest"])
    stress = IndependentChangeDataset(config["data"]["stress_val_manifest"])

    model = TerraDeltaSiameseV2(**config["model"]).to(device)
    initialized = initialize_encoder_from_baseline(model, config["training"]["baseline_checkpoint"])

    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder = [
        parameter for parameter in model.parameters()
        if id(parameter) in encoder_ids and parameter.requires_grad
    ]
    new_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in encoder_ids and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": float(config["training"]["encoder_lr"])},
            {"params": new_parameters, "lr": float(config["training"]["new_lr"])},
        ],
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    loss_fn = V2Loss(**config["training"]["loss"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    milestones = set(map(int, config["training"]["checkpoints"]))
    max_steps = max(milestones)
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    groups = pools(train)
    history = []
    started = time.time()

    for step in range(1, max_steps + 1):
        model.train()
        indices = draw_batch(groups, int(config["training"]["batch_size"]), rng)
        samples = [
            train.get(index, seed=seed + step * 10007 + offset)
            for offset, index in enumerate(indices)
        ]
        batch = collate(samples, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(batch["image"])
            loss, parts = loss_fn(
                outputs,
                batch["target"],
                batch["valid_mask"],
                batch["presence"],
                batch["presence_valid"],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"].get("max_grad_norm", 1.0))
        )
        scaler.step(optimizer)
        scaler.update()

        if step in milestones:
            checkpoint = output_dir / f"step-{step:04d}.pt"
            save_v2_checkpoint(
                checkpoint, model, steps=step, seed=seed, initialization=initialized
            )
            legacy_score = evaluate(model, legacy, device, config)
            stress_score = evaluate(model, stress, device, config)
            item = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "parts": {key: float(value.cpu()) for key, value in parts.items()},
                "legacy": legacy_score,
                "stress": stress_score,
                "elapsed_seconds": time.time() - started,
            }
            history.append(item)
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2, allow_nan=False)
            )
            print(json.dumps(item, allow_nan=False), flush=True)

    result = {
        "device": str(device),
        "train_samples": len(train),
        "legacy_val": len(legacy),
        "stress_val": len(stress),
        "pools": {key: len(value) for key, value in groups.items()},
        "init": {
            "copied": len(initialized["copied"]),
            "converted": initialized["converted"],
            "skipped": len(initialized["skipped"]),
        },
        "history": history,
        "runtime_seconds": time.time() - started,
    }
    (output_dir / "run-summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
