#!/usr/bin/env python3
"""TerraDelta V3 Satlas Swin-v2 Training Engine.

Features:
- Siamese Satlas Swin-v2-Base backbone loaded from official Satlas checkpoint
- Multi-scale directional temporal fusion + FPN decoder
- Class-specific masked BCE + Dice loss
- STRICT AIHub tree masking: valid_mask_tree = 0, presence_valid_tree = 0
- Differential learning rates: 1e-5 for backbone, 1e-4 for heads
- Mixed precision (BF16/FP16)
- Checkpointing at steps 200, 400, 600, 800, 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terradelta.models.satlas_v3 import SatlasV3Model


CLASSES = ("new_building", "tree_removal")


class SatlasV3Dataset(Dataset):
    """Dataset for TerraDelta V3 Siamese training with Satlas conventions."""

    def __init__(self, manifest_path: str | Path, is_train: bool = True):
        self.manifest_path = Path(manifest_path).resolve()
        self.is_train = is_train

        with self.manifest_path.open(newline="", encoding="utf-8-sig") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

        # Resolve paths relative to manifest if relative
        for r in self.rows:
            for key in ("pre", "post", "new_building", "tree_removal", "valid_mask", "review_mask"):
                val = (r.get(key) or "").strip()
                if val and val.lower() != "absent":
                    p = Path(val)
                    r[key] = str(p if p.is_absolute() else (self.manifest_path.parent / p).resolve())

    def __len__(self) -> int:
        return len(self.rows)

    def _read_image(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"), dtype=np.uint8)

    def _read_mask(self, path: str, shape: tuple[int, int]) -> np.ndarray:
        if not path or path.lower() == "absent" or not Path(path).is_file():
            return np.zeros(shape, dtype=np.float32)
        with Image.open(path) as im:
            return (np.array(im.convert("L")) > 0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        pre = self._read_image(row["pre"])
        post = self._read_image(row["post"])
        h, w = pre.shape[:2]

        # Target masks
        bldg_mask = self._read_mask(row.get("new_building", ""), (h, w))
        tree_mask = self._read_mask(row.get("tree_removal", ""), (h, w))
        target = np.stack([bldg_mask, tree_mask], axis=0)  # (2, H, W)

        # Valid masks
        valid_bldg = np.ones((h, w), dtype=np.float32)
        valid_tree = np.ones((h, w), dtype=np.float32)

        # Presence valid flags
        pres_valid = np.ones(2, dtype=np.float32)

        # Check AIHub or explicit flags
        is_aihub = (
            row.get("source") == "aihub_71363"
            or "aihub" in row.get("category", "").lower()
            or str(row.get("valid_mask_tree", "")).lower() in ("false", "0")
        )

        if is_aihub:
            # STRICT POLICY: Tree loss is fully masked on AIHub samples
            valid_tree[:] = 0.0
            pres_valid[1] = 0.0

        # Review mask or partial annotation handling for TerraDelta data
        rev_path = row.get("review_mask", "")
        if rev_path and Path(rev_path).is_file():
            rev = self._read_mask(rev_path, (h, w))
            valid_bldg *= rev
            valid_tree *= rev

        valid_mask = np.stack([valid_bldg, valid_tree], axis=0)  # (2, H, W)

        # Presence labels: 1.0 if class appears, else 0.0
        presence = np.array([
            float(bldg_mask.sum() > 0),
            float(tree_mask.sum() > 0),
        ], dtype=np.float32)

        # Augmentation during training
        if self.is_train:
            # Random horizontal flip
            if random.random() < 0.5:
                pre = np.fliplr(pre)
                post = np.fliplr(post)
                target = np.flip(target, axis=2)
                valid_mask = np.flip(valid_mask, axis=2)

            # Random vertical flip
            if random.random() < 0.5:
                pre = np.flipud(pre)
                post = np.flipud(post)
                target = np.flip(target, axis=1)
                valid_mask = np.flip(valid_mask, axis=1)

            # Random 90-degree rotations
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                pre = np.rot90(pre, rot_k, axes=(0, 1))
                post = np.rot90(post, rot_k, axes=(0, 1))
                target = np.rot90(target, rot_k, axes=(1, 2))
                valid_mask = np.rot90(valid_mask, rot_k, axes=(1, 2))

            # Light photometric jitter
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-15, 15)
                pre = np.clip(pre.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                post = np.clip(post.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Normalization: strictly [0, 1] float RGB, matching SatlasPretrain
        pre_t = torch.from_numpy(np.ascontiguousarray(pre.transpose(2, 0, 1))).float() / 255.0
        post_t = torch.from_numpy(np.ascontiguousarray(post.transpose(2, 0, 1))).float() / 255.0
        image = torch.cat([pre_t, post_t], dim=0)  # (6, H, W)

        return {
            "id": row["id"],
            "image": image,
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)),
            "presence": torch.from_numpy(presence),
            "presence_valid": torch.from_numpy(pres_valid),
        }


class MaskedV3Loss(nn.Module):
    """Loss module with strict class-level masking for segmentation and presence."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        presence_weight: float = 0.25,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.presence_weight = presence_weight
        self.smooth = smooth

    def forward(
        self,
        seg_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        presence: torch.Tensor,
        presence_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Seg BCE
        bce_raw = F.binary_cross_entropy_with_logits(seg_logits, target, reduction="none")
        bce_masked = bce_raw * valid_mask
        bce_loss = bce_masked.sum() / valid_mask.sum().clamp_min(1.0)

        # Seg Dice per class
        prob = torch.sigmoid(seg_logits) * valid_mask
        truth = target * valid_mask
        axes = (0, 2, 3)
        intersection = (prob * truth).sum(dim=axes)
        cardinality = prob.sum(dim=axes) + truth.sum(dim=axes)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Compute dice loss only for classes that have valid pixels in the batch
        class_valid_pixels = valid_mask.sum(dim=(0, 2, 3)) > 0
        if class_valid_pixels.any():
            dice_loss = (1.0 - dice[class_valid_pixels]).mean()
        else:
            dice_loss = torch.tensor(0.0, device=seg_logits.device)

        # Presence BCE
        p_raw = F.binary_cross_entropy_with_logits(presence_logits, presence, reduction="none")
        p_masked = p_raw * presence_valid
        presence_loss = p_masked.sum() / presence_valid.sum().clamp_min(1.0)

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.presence_weight * presence_loss
        )

        metrics = {
            "total_loss": total_loss.item(),
            "bce": bce_loss.item(),
            "dice": dice_loss.item(),
            "presence": presence_loss.item(),
        }
        return total_loss, metrics


def train_recipe(
    recipe_name: str,
    manifest_path: str | Path,
    satlas_weights: str | Path,
    output_dir: str | Path,
    total_steps: int = 1000,
    batch_size: int = 8,
    lr_backbone: float = 1e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 1e-2,
    save_interval: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute training for a single recipe (S0, S1, or S2)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Starting {recipe_name} Training on {device}")
    print(f"Manifest: {manifest_path}")
    print(f"Satlas weights: {satlas_weights}")
    print(f"Steps: {total_steps}, Batch size: {batch_size}")
    print(f"=======================================================")

    out_path = Path(output_dir) / recipe_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = SatlasV3Model(
        weights_path=satlas_weights,
        fpn_channels=128,
        num_classes=2,
        freeze_backbone=False,
    )
    model.to(device)

    # Dataset & Loader
    dataset = SatlasV3Dataset(manifest_path, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer with differential learning rates
    backbone_params = list(model.backbone.parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": other_params, "lr": lr_heads},
        ],
        weight_decay=weight_decay,
    )

    loss_fn = MaskedV3Loss(bce_weight=1.0, dice_weight=1.0, presence_weight=0.25)

    # Cosine scheduler with 50-step linear warmup
    def lr_lambda(step: int) -> float:
        warmup_steps = 50
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP Autocast
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    step = 0
    epoch = 0
    t0 = time.time()
    step_history = []

    model.train()
    while step < total_steps:
        epoch += 1
        for batch in loader:
            step += 1
            if step > total_steps:
                break

            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            presence = batch["presence"].to(device, non_blocking=True)
            presence_valid = batch["presence_valid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                seg_logits, presence_logits = model(images)
                loss, metrics = loss_fn(
                    seg_logits, presence_logits, target, valid_mask, presence, presence_valid
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            if step % 20 == 0 or step == 1:
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                print(
                    f"[{recipe_name}] Step {step}/{total_steps} (epoch {epoch}) | "
                    f"Loss: {metrics['total_loss']:.4f} (bce: {metrics['bce']:.4f}, dice: {metrics['dice']:.4f}, pres: {metrics['presence']:.4f}) | "
                    f"Speed: {steps_per_sec:.1f} step/s"
                )

            if step % 50 == 0:
                step_history.append({"step": step, **metrics})

            # Checkpoint at intervals
            if step % save_interval == 0 or step == total_steps:
                ckpt_file = out_path / f"checkpoint_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "recipe": recipe_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": metrics["total_loss"],
                    },
                    ckpt_file,
                )
                print(f"[{recipe_name}] Saved checkpoint to {ckpt_file}")

    total_time = time.time() - t0
    final_ckpt = out_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "recipe": recipe_name}, final_ckpt)
    print(f"[{recipe_name}] Completed {total_steps} steps in {total_time:.1f}s ({total_time/60:.1f} min). Saved {final_ckpt}")

    summary = {
        "recipe": recipe_name,
        "manifest": str(manifest_path),
        "steps": total_steps,
        "total_time_sec": round(total_time, 2),
        "final_loss": metrics["total_loss"],
        "final_checkpoint": str(final_ckpt),
        "history": step_history,
    }
    with (out_path / "train_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train TerraDelta V3 Satlas Models")
    parser.add_argument("--recipe", type=str, required=True, choices=["S0", "S1", "S2", "all"])
    parser.add_argument("--manifests-dir", type=str, default="/opt/dlami/nvme/data/v3_manifests")
    parser.add_argument("--satlas-weights", type=str, default="/opt/dlami/nvme/models/satlas/aerial_swinb_si.pth")
    parser.add_argument("--output-dir", type=str, default="/opt/dlami/nvme/outputs/v3_satlas")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=200)
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)

    recipes = ["S0", "S1", "S2"] if args.recipe == "all" else [args.recipe]
    for r in recipes:
        manifest_file = manifests_dir / f"train_{r}.csv"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")
        train_recipe(
            recipe_name=r,
            manifest_path=manifest_file,
            satlas_weights=args.satlas_weights,
            output_dir=args.output_dir,
            total_steps=args.steps,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
        )


if __name__ == "__main__":
    main()
