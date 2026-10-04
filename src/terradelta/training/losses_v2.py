"""Independent binary segmentation and pair-presence loss for TerraDelta v2."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class V2Loss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, presence_weight=.25,
                 focal_gamma=0.0, smooth=1e-6):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.presence_weight = float(presence_weight)
        self.focal_gamma = float(focal_gamma)
        self.smooth = float(smooth)

    def forward(self, outputs, target, valid_mask, presence, presence_valid):
        logits = outputs["segmentation"].float()
        p_logits = outputs["presence"].float()
        if logits.shape != target.shape or logits.shape[1] != 2:
            raise ValueError("segmentation/target must be B2HW")
        valid = valid_mask[:, None].expand_as(target)
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.focal_gamma > 0:
            prob = torch.sigmoid(logits)
            pt = torch.where(target > .5, prob, 1 - prob)
            raw = raw * (1 - pt).pow(self.focal_gamma)
        denom = valid.sum().clamp_min(1)
        bce = (raw * valid).sum() / denom
        prob = torch.sigmoid(logits) * valid
        truth = target * valid
        axes = (0, 2, 3)
        intersection = (prob * truth).sum(axes)
        dice = (2 * intersection + self.smooth) / (prob.sum(axes) + truth.sum(axes) + self.smooth)
        dice_loss = (1 - dice).mean()
        p_raw = F.binary_cross_entropy_with_logits(p_logits, presence.float(), reduction="none")
        pv = presence_valid.float()
        p_loss = (p_raw * pv).sum() / pv.sum().clamp_min(1)
        total = self.bce_weight * bce + self.dice_weight * dice_loss + self.presence_weight * p_loss
        return total, {"bce": bce.detach(), "dice_loss": dice_loss.detach(), "presence_loss": p_loss.detach()}
