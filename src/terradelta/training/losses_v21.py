"""Two supervised v2 views with masked photometric probability consistency."""
from __future__ import annotations

import math

import torch
from torch import nn

from terradelta.training.losses_v2 import V2Loss


class V21Loss(nn.Module):
    """Apply after two SiameseV2 forwards with identical geometry and labels.

    ``consistency_weight`` is bounded to [0, 0.20]; intended comparisons use
    0.05, 0.10, or 0.20. Zero disables the consistency contribution.
    Pixel validity is BHW, or the identical-channel B2HW expansion of that
    shared mask required by V2Loss. Presence labels and validity are B2.
    Both views receive the full supervised objective, including positives.
    """

    def __init__(self, consistency_weight=0.05, *, bce_weight=1.0, dice_weight=1.0,
                 presence_weight=0.25, focal_gamma=0.0, smooth=1e-6):
        super().__init__()
        weight = float(consistency_weight)
        if not math.isfinite(weight) or not 0.0 <= weight <= 0.20:
            raise ValueError("consistency_weight must be finite and in [0, 0.20]")
        self.consistency_weight = weight
        self.supervised_loss = V2Loss(
            bce_weight=bce_weight, dice_weight=dice_weight,
            presence_weight=presence_weight, focal_gamma=focal_gamma, smooth=smooth,
        )

    def forward(self, outputs, photometric_outputs, target, valid_mask, presence, presence_valid):
        """Return differentiable total and detached scalar diagnostic parts.

        Consistency is the sum of segmentation and presence probability MSE,
        each normalized by its own number of valid elements. An empty mask
        contributes zero and retains gradient connections to both views.
        """
        if target.ndim != 4 or target.shape[1] != 2:
            raise ValueError("target must be B2HW")
        batch, _, height, width = target.shape
        if presence.shape != (batch, 2):
            raise ValueError("presence must be B2")
        if presence_valid.shape != (batch, 2):
            raise ValueError("presence_valid must be B2")
        if valid_mask.shape == target.shape:
            if not torch.equal(valid_mask[:, 0], valid_mask[:, 1]):
                raise ValueError("B2HW valid_mask must have identical channels for V2Loss")
            valid_mask = valid_mask[:, 0]
        elif valid_mask.shape != (batch, height, width):
            raise ValueError("valid_mask must be BHW or its identical-channel B2HW expansion")
        for name, view in (("outputs", outputs), ("photometric_outputs", photometric_outputs)):
            if view["segmentation"].shape != target.shape:
                raise ValueError(f"{name} segmentation logits must match B2HW target")
            if view["presence"].shape != presence.shape:
                raise ValueError(f"{name} presence logits must match B2 presence")

        target = target.float()
        original_loss, original_parts = self.supervised_loss(
            outputs, target, valid_mask, presence, presence_valid,
        )
        photometric_loss, photometric_parts = self.supervised_loss(
            photometric_outputs, target, valid_mask, presence, presence_valid,
        )
        supervised = (original_loss + photometric_loss) * 0.5

        pixel_valid = valid_mask[:, None].expand_as(target).float()
        pair_valid = presence_valid.float()
        segmentation_mse = (
            (torch.sigmoid(outputs["segmentation"].float())
             - torch.sigmoid(photometric_outputs["segmentation"].float())).square() * pixel_valid
        ).sum() / pixel_valid.sum().clamp_min(1)
        presence_mse = (
            (torch.sigmoid(outputs["presence"].float())
             - torch.sigmoid(photometric_outputs["presence"].float())).square() * pair_valid
        ).sum() / pair_valid.sum().clamp_min(1)
        consistency = segmentation_mse + presence_mse
        weighted_consistency = self.consistency_weight * consistency
        total = supervised + weighted_consistency
        parts = {
            key: (value + photometric_parts[key]) * 0.5
            for key, value in original_parts.items()
        }
        parts.update({
            "supervised_original": original_loss.detach(),
            "supervised_photometric": photometric_loss.detach(),
            "supervised": supervised.detach(),
            "consistency_segmentation": segmentation_mse.detach(),
            "consistency_presence": presence_mse.detach(),
            "consistency": consistency.detach(),
            "weighted_consistency": weighted_consistency.detach(),
            "total": total.detach(),
        })
        return total, parts
