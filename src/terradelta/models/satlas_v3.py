"""TerraDelta V3: SatlasPretrain Aerial Swin-v2-Base Siamese Architecture.

Implements shared-weight Siamese temporal change detection model:
- Shared Satlas Aerial Swin-v2-Base encoder for PRE and POST RGB imagery
- Multi-scale directional temporal fusion: [PRE, POST, abs(POST - PRE), POST - PRE]
- Top-Down FPN feature decoder
- Independent segmentation heads: new_building and tree_removal
- Independent pair-level presence heads: new_building and tree_removal
- Fully offline weight loading and execution compatible with A10G and CPU
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def adjust_state_dict_prefix(state_dict: dict[str, torch.Tensor], needed: str, prefix: str | None = None, prefix_allowed_count: int | None = None) -> dict[str, torch.Tensor]:
    """Adjust state dict prefixes from Satlas checkpoint."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if needed not in key:
            continue
        if prefix is not None and prefix_allowed_count is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, "", 1)
        new_state_dict[key] = value
    return new_state_dict


class SwinV2Backbone(nn.Module):
    """Swin-v2-Base backbone yielding multi-scale feature maps."""
    def __init__(self, num_channels: int = 3, arch: str = "swinb"):
        super().__init__()
        if arch == "swinb":
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [128, 256, 512, 1024]
        elif arch == "swint":
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [96, 192, 384, 768]
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        if num_channels != 3:
            orig_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                num_channels, orig_conv.out_channels, kernel_size=orig_conv.kernel_size, stride=orig_conv.stride
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale feature maps from Swin-v2."""
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2).contiguous())
        # Return 4 scales: strides 4, 8, 16, 32
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class DirectionalScaleFusion(nn.Module):
    """Directional temporal fusion [PRE, POST, abs(POST - PRE), POST - PRE] + 1x1 projection."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Input channels is 4 * in_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(post - pre)
        dir_diff = post - pre
        fused = torch.cat([pre, post, abs_diff, dir_diff], dim=1)
        return self.proj(fused)


class SatlasV3Model(nn.Module):
    """TerraDelta V3 Satlas Swin-v2 Siamese Model."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        arch: str = "swinb",
        fpn_channels: int = 128,
        num_classes: int = 2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels

        # Shared Siamese Backbone
        self.backbone = SwinV2Backbone(num_channels=3, arch=arch)
        channels = self.backbone.out_channels  # [128, 256, 512, 1024]

        # Multi-scale directional fusion projections
        self.fusions = nn.ModuleList([
            DirectionalScaleFusion(c, fpn_channels) for c in channels
        ])

        # Lateral FPN smoothing
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.GELU(),
            )
            for _ in channels
        ])

        # Aggregation of all scales at stride 4 (64x64)
        self.agg_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels * 2),
            nn.GELU(),
        )

        # Upsample decoder to stride 1 (256x256)
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(fpn_channels * 2, fpn_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(fpn_channels, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Independent Segmentation Head (num_classes logits)
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Independent Presence Head
        # Global pooling from deepest feature scales
        self.presence_head = nn.Sequential(
            nn.Linear(fpn_channels * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        if weights_path is not None:
            self.load_satlas_weights(weights_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def load_satlas_weights(self, weights_path: str | Path) -> None:
        """Load official SatlasPretrain weights into backbone."""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Satlas checkpoint not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        # Extract backbone keys
        if "backbone" in str(state_dict.keys()):
            adjusted = adjust_state_dict_prefix(state_dict, "backbone", "backbone.", 1)
        else:
            adjusted = state_dict
        msg = self.backbone.load_state_dict(adjusted, strict=False)
        print(f"Loaded Satlas pretrained weights from {weights_path.name}: {len(adjusted)} keys loaded.")

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """Forward pass for 6-channel PRE+POST tensor of shape (B, 6, H, W)."""
        pre_rgb = images[:, :3]
        post_rgb = images[:, 3:6]

        # Shared backbone forward
        pre_feats = self.backbone(pre_rgb)
        post_feats = self.backbone(post_rgb)

        # Multi-scale directional fusion
        f0 = self.fusions[0](pre_feats[0], post_feats[0])  # stride 4 (64x64)
        f1 = self.fusions[1](pre_feats[1], post_feats[1])  # stride 8 (32x32)
        f2 = self.fusions[2](pre_feats[2], post_feats[2])  # stride 16 (16x16)
        f3 = self.fusions[3](pre_feats[3], post_feats[3])  # stride 32 (8x8)

        # Top-down lateral path
        p3 = f3
        p2 = f2 + F.interpolate(p3, size=f2.shape[2:], mode="nearest")
        p1 = f1 + F.interpolate(p2, size=f1.shape[2:], mode="nearest")
        p0 = f0 + F.interpolate(p1, size=f0.shape[2:], mode="nearest")

        # Smooth lateral features
        s3 = self.smooth[3](p3)
        s2 = self.smooth[2](p2)
        s1 = self.smooth[1](p1)
        s0 = self.smooth[0](p0)

        # Aggregate all to stride 4 (64x64)
        s3_up = F.interpolate(s3, size=s0.shape[2:], mode="bilinear", align_corners=False)
        s2_up = F.interpolate(s2, size=s0.shape[2:], mode="bilinear", align_corners=False)
        s1_up = F.interpolate(s1, size=s0.shape[2:], mode="bilinear", align_corners=False)
        agg = self.agg_conv(torch.cat([s0, s1_up, s2_up, s3_up], dim=1))

        # Decode to stride 1 (256x256)
        dec = self.decoder(agg)
        seg_logits = self.seg_head(dec)

        # Presence head pooling from deepest levels (p3 and p2)
        p3_avg = F.adaptive_avg_pool2d(p3, 1).flatten(1)
        p3_max = F.adaptive_max_pool2d(p3, 1).flatten(1)
        p2_avg = F.adaptive_avg_pool2d(p2, 1).flatten(1)
        p2_max = F.adaptive_max_pool2d(p2, 1).flatten(1)
        presence_feats = torch.cat([p3_avg, p3_max, p2_avg, p2_max], dim=1)
        presence_logits = self.presence_head(presence_feats)

        return seg_logits, presence_logits


class V3SatlasPredictor:
    """Inference wrapper for Satlas V3 model matching V2Predictor interface."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Mapping[str, Any],
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.config = dict(config)
        self.model = SatlasV3Model(weights_path=None, fpn_channels=128, num_classes=2)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def probabilities(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pixel and presence probabilities in [0, 1]."""
        images = images.to(self.device, dtype=torch.float32)
        seg_logits, presence_logits = self.model(images)
        pixels = torch.sigmoid(seg_logits).cpu()
        presence = torch.sigmoid(presence_logits).cpu()
        return pixels, presence
