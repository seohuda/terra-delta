"""TerraDelta v2: shared Siamese ResNet18 with temporal fusion and light alignment."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FlowAlign(nn.Module):
    """Predict a bounded offset field and warp POST features toward PRE."""

    def __init__(self, channels, max_offset=1.5):
        super().__init__()
        hidden = max(16, channels // 4)
        self.max_offset = float(max_offset)
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, pre, post):
        if pre.shape != post.shape:
            raise ValueError("alignment expects matched feature maps")
        b, _, h, w = pre.shape
        flow = torch.tanh(self.offset(torch.cat([pre, post], 1))) * self.max_offset
        ys = torch.linspace(-1, 1, h, device=pre.device, dtype=pre.dtype)
        xs = torch.linspace(-1, 1, w, device=pre.device, dtype=pre.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(b, -1, -1, -1).clone()
        if w > 1:
            grid[..., 0] += 2.0 * flow[:, 0] / (w - 1)
        if h > 1:
            grid[..., 1] += 2.0 * flow[:, 1] / (h - 1)
        return F.grid_sample(post, grid, mode="bilinear", padding_mode="border", align_corners=True)


class TemporalFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre, post):
        return self.reduce(torch.cat([pre, post, (post - pre).abs(), pre * post], 1))


class TerraDeltaSiameseV2(nn.Module):
    """Independent building/tree masks plus pair-level presence logits."""

    architecture = "siamese_v2"

    def __init__(self, *, alignment=True, alignment_scales=(3, 4), max_offset=1.5):
        super().__init__()
        self.encoder = smp.encoders.get_encoder("resnet18", in_channels=3, depth=5, weights=None)
        channels = list(self.encoder.out_channels[1:])
        self.aligners = nn.ModuleDict({
            str(i): FlowAlign(channels[i - 1], max_offset=max_offset)
            for i in alignment_scales if alignment and 1 <= i <= 5
        })
        self.fusions = nn.ModuleList([TemporalFusion(c) for c in channels])
        self.dec4 = ConvBlock(channels[4] + channels[3], 256)
        self.dec3 = ConvBlock(256 + channels[2], 128)
        self.dec2 = ConvBlock(128 + channels[1], 64)
        self.dec1 = ConvBlock(64 + channels[0], 64)
        self.segmentation_head = nn.Conv2d(64, 2, 1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], 2)
        )

    def _fuse(self, pre_features, post_features):
        fused = []
        for i, (pre, post, fuse) in enumerate(zip(pre_features[1:], post_features[1:], self.fusions), start=1):
            if str(i) in self.aligners:
                post = self.aligners[str(i)](pre, post)
            fused.append(fuse(pre, post))
        return fused

    def forward(self, image, post=None):
        if post is None:
            if image.ndim != 4 or image.shape[1] != 6:
                raise ValueError("TerraDeltaSiameseV2 expects B6HW or separate B3HW tensors")
            pre, post = image[:, :3], image[:, 3:]
        else:
            pre = image
            if pre.ndim != 4 or post.ndim != 4 or pre.shape != post.shape or pre.shape[1] != 3:
                raise ValueError("separate inputs must be matched B3HW tensors")
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        f1, f2, f3, f4, f5 = self._fuse(pre_features, post_features)
        presence = self.presence_head(f5)
        x = self.dec4(torch.cat([
            F.interpolate(f5, size=f4.shape[-2:], mode="bilinear", align_corners=False), f4
        ], 1))
        x = self.dec3(torch.cat([
            F.interpolate(x, size=f3.shape[-2:], mode="bilinear", align_corners=False), f3
        ], 1))
        x = self.dec2(torch.cat([
            F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2
        ], 1))
        x = self.dec1(torch.cat([
            F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1
        ], 1))
        seg = self.segmentation_head(
            F.interpolate(x, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        )
        return {"segmentation": seg, "presence": presence}


def initialize_encoder_from_baseline(model, checkpoint):
    """Copy organizer ResNet18 encoder weights; average the two RGB halves of conv1."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError("invalid baseline checkpoint")
    target = model.encoder.state_dict()
    copied, converted, skipped = [], [], []
    for key, value in state.items():
        if not key.startswith("encoder."):
            continue
        name = key[len("encoder."):]
        if name not in target:
            skipped.append(name)
            continue
        if name == "conv1.weight" and value.ndim == 4 and value.shape[1] == 6 and target[name].shape[1] == 3:
            target[name] = 0.5 * (value[:, :3] + value[:, 3:6])
            converted.append(name)
        elif value.shape == target[name].shape:
            target[name] = value
            copied.append(name)
        else:
            skipped.append(name)
    model.encoder.load_state_dict(target, strict=True)
    return {"copied": copied, "converted": converted, "skipped": skipped}


def save_v2_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": "siamese_v2",
        "encoder": "resnet18",
        "input": "pre_rgb+post_rgb",
        "classes": ["new_building", "tree_removal"],
        "state_dict": model.state_dict(),
        **metadata,
    }
    torch.save(payload, path)


def load_v2_checkpoint(path, model=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise ValueError("v2 checkpoint must contain state_dict")
    if payload.get("architecture") != "siamese_v2":
        raise ValueError("not a TerraDelta v2 checkpoint")
    if payload.get("encoder") != "resnet18" or payload.get("input") != "pre_rgb+post_rgb":
        raise ValueError("v2 checkpoint input/encoder contract is incompatible")
    if payload.get("classes") != ["new_building", "tree_removal"]:
        raise ValueError("v2 checkpoint class order is incompatible")
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("invalid v2 state_dict")
    if model is not None:
        model.load_state_dict(payload["state_dict"], strict=True)
    return {k: v for k, v in payload.items() if k != "state_dict"}
