"""Configurable multiclass losses; targets use 0/1/2 class labels."""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    """CE, weighted CE, soft Dice, Focal, and additive combinations."""

    def __init__(self, name="ce_dice", class_weights=None, gamma=2.0,
                 ce_weight=1.0, dice_weight=1.0, include_background=False,
                 ignore_index=-100, smooth=1e-6):
        super().__init__()
        aliases = {"crossentropy": "ce", "cross_entropy": "ce",
                   "weighted_crossentropy": "weighted_ce", "weighted_cross_entropy": "weighted_ce",
                   "ce+dice": "ce_dice", "focal+dice": "focal_dice"}
        self.name = aliases.get(str(name).lower(), str(name).lower())
        if self.name not in {"ce", "weighted_ce", "dice", "ce_dice", "focal", "focal_dice"}:
            raise ValueError(f"Unknown loss: {name}")
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("class_weights must contain three finite nonnegative weights with positive sum")
        else:
            weights = None
        if self.name == "weighted_ce" and weights is None:
            raise ValueError("weighted_ce requires class_weights")
        self.register_buffer("class_weights", weights)
        self.gamma, self.ce_weight, self.dice_weight = float(gamma), float(ce_weight), float(dice_weight)
        self.smooth = float(smooth)
        if any(not math.isfinite(v) or v < 0 for v in (self.gamma, self.ce_weight, self.dice_weight)) or not math.isfinite(self.smooth) or self.smooth <= 0:
            raise ValueError("Loss coefficients must be finite and nonnegative; smooth must be positive")
        self.include_background, self.ignore_index = bool(include_background), int(ignore_index)

    def forward(self, logits, target):
        if logits.ndim != 4 or logits.shape[1] != 3 or target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError("Expected logits B3HW and target BHW")
        if target.dtype != torch.long:
            raise ValueError("Target must be torch.long class indices")
        valid = target != self.ignore_index
        if not valid.any():
            return logits.float().sum() * 0.0
        if ((target[valid] < 0) | (target[valid] > 2)).any():
            raise ValueError("Target labels must be 0, 1, 2 or ignore_index")
        # Reductions stay in float32 even under autocast.
        logits = logits.float()
        result = logits.sum() * 0.0
        if self.name in {"ce", "weighted_ce", "ce_dice"}:
            ce = F.cross_entropy(logits, target, weight=self.class_weights,
                                 ignore_index=self.ignore_index, reduction="none")
            denom = valid.sum() if self.class_weights is None else self.class_weights[target[valid]].sum()
            result = self.ce_weight * ce[valid].sum() / denom.clamp_min(self.smooth)
        if self.name in {"focal", "focal_dice"}:
            log_probs = F.log_softmax(logits, dim=1)
            safe = target.masked_fill(~valid, 0)
            log_pt = log_probs.gather(1, safe.unsqueeze(1)).squeeze(1)
            focal = -(1.0 - log_pt.exp()).pow(self.gamma) * log_pt
            if self.class_weights is not None:
                focal = focal * self.class_weights[safe]
            result = self.ce_weight * focal[valid].mean()
        if self.name in {"dice", "ce_dice", "focal_dice"}:
            safe = target.masked_fill(~valid, 0)
            onehot = F.one_hot(safe, num_classes=3).permute(0, 3, 1, 2).float()
            keep = valid.unsqueeze(1)
            probs, onehot = logits.softmax(1) * keep, onehot * keep
            axes = (0, 2, 3)
            overlap = (probs * onehot).sum(axes)
            dice = (2 * overlap + self.smooth) / (probs.sum(axes) + onehot.sum(axes) + self.smooth)
            start = 0 if self.include_background else 1
            per_class = 1 - dice[start:]
            if self.class_weights is None:
                value = per_class.mean()
            else:
                weights = self.class_weights[start:]
                if weights.sum() <= 0:
                    raise ValueError("Selected Dice classes need positive total class weight")
                value = (per_class * weights).sum() / weights.sum()
            result = result + self.dice_weight * value
        return result


def build_loss(config=None):
    """Accept a full config, a training section, a loss mapping, or a name."""
    config = config or {}
    if isinstance(config, str):
        return SegmentationLoss(config)
    if not isinstance(config, Mapping):
        raise TypeError("Loss configuration must be a mapping or name")
    section = config.get("training", config)
    loss_keys = {"name", "class_weights", "gamma", "ce_weight", "dice_weight", "include_background", "ignore_index", "smooth"}
    loss = section.get("loss", section if set(section) <= loss_keys else {})
    if isinstance(loss, str):
        return SegmentationLoss(loss)
    return SegmentationLoss(**dict(loss))
