"""Paired geometry and independent appearance changes without global RNG state."""

from __future__ import annotations

import cv2
import numpy as np


def photometric(image, rng, probability=0.35):
    """Brightness, contrast, gamma, saturation, hue, blur, sharpen, JPEG and haze."""
    result = image.copy()
    if rng.random() < probability:
        contrast, brightness = rng.uniform(0.85, 1.15), rng.uniform(-20, 20)
        result = np.clip((result.astype(np.float32) - 127.5) * contrast + 127.5 + brightness, 0, 255).astype(np.uint8)
    if rng.random() < probability:
        gamma = rng.uniform(0.85, 1.2)
        lut = np.clip((np.arange(256) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
        result = cv2.LUT(result, lut)
    if rng.random() < probability:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < probability:
        result = cv2.GaussianBlur(result, (3, 3), rng.uniform(0.2, 1.0))
    if rng.random() < probability:
        lowpass = cv2.GaussianBlur(result, (3, 3), 0.8)
        result = cv2.addWeighted(result, 1.3, lowpass, -0.3, 0)
    if rng.random() < probability:
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        if not success:
            raise RuntimeError("JPEG augmentation encoding failed")
        result = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if rng.random() < probability:
        haze = rng.uniform(0.01, 0.12)
        result = np.clip(result.astype(np.float32) * (1 - haze) + 235 * haze, 0, 255).astype(np.uint8)
    return result


class PairedTransform:
    """Identical geometry for both timestamps and nearest-neighbor label warps.

    ``seed`` initializes a local, reproducible sequence; an explicit call-time
    seed replays one augmentation. Recreate with distinct seeds for each data
    loader worker (or reseed via ``reseed``). No global numpy/Python state changes.
    """

    def __init__(self, seed=0, training=True, geometric=True, photometric_probability=0.35,
                 affine_probability=0.5, max_translation=0.03, max_rotation=5.0, max_scale=0.05):
        if not 0 <= photometric_probability <= 1 or not 0 <= affine_probability <= 1:
            raise ValueError("Augmentation probabilities must be in [0, 1]")
        if not 0 <= max_scale < 1 or not 0 <= max_translation < 1 or max_rotation < 0:
            raise ValueError("Invalid affine limits")
        self.rng = np.random.default_rng(seed)
        self.training, self.geometric = training, geometric
        self.photo_p, self.affine_p = photometric_probability, affine_probability
        self.translation, self.rotation, self.scale = max_translation, max_rotation, max_scale

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, *, pre, post, mask, seed=None):
        if pre.dtype != np.uint8 or post.dtype != np.uint8:
            raise ValueError("PairedTransform expects uint8 RGB imagery")
        if pre.shape != post.shape or mask.shape != pre.shape[:2]:
            raise ValueError("PairedTransform inputs have different shapes")
        pre, post, mask = pre.copy(), post.copy(), mask.copy()
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        if self.training and self.geometric:
            if rng.random() < 0.5:
                pre, post, mask = np.flip(pre, 1), np.flip(post, 1), np.flip(mask, 1)
            if rng.random() < 0.5:
                pre, post, mask = np.flip(pre, 0), np.flip(post, 0), np.flip(mask, 0)
            turns = int(rng.integers(0, 4))
            # A rectangular input cannot swap dimensions in a batch.
            if pre.shape[0] != pre.shape[1]:
                turns = 2 * (turns % 2)
            pre, post, mask = np.rot90(pre, turns), np.rot90(post, turns), np.rot90(mask, turns)
            if rng.random() < self.affine_p:
                height, width = mask.shape
                matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2),
                                                 rng.uniform(-self.rotation, self.rotation),
                                                 rng.uniform(1 - self.scale, 1 + self.scale))
                matrix[:, 2] += rng.uniform(-self.translation, self.translation, 2) * [width, height]
                # Reflect all three identically; never synthesize unlabeled black edges.
                opts = dict(dsize=(width, height), borderMode=cv2.BORDER_REFLECT_101)
                pre = cv2.warpAffine(pre, matrix, flags=cv2.INTER_LINEAR, **opts)
                post = cv2.warpAffine(post, matrix, flags=cv2.INTER_LINEAR, **opts)
                mask = cv2.warpAffine(mask, matrix, flags=cv2.INTER_NEAREST, **opts)
        if self.training and self.photo_p:
            pre, post = photometric(pre, rng, self.photo_p), photometric(post, rng, self.photo_p)
        return {"pre": np.ascontiguousarray(pre), "post": np.ascontiguousarray(post),
                "mask": np.ascontiguousarray(mask)}


def build_transforms(config=None, *, training=True, seed=0):
    """Accept either augmentation options or a full config with ``augmentation``."""
    config = config or {}
    options = dict(config.get("augmentation", config))
    enabled = options.pop("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("augmentation.enabled must be a boolean")
    options.setdefault("seed", seed)
    options["training"] = bool(options.get("training", training)) and enabled
    return PairedTransform(**options)
