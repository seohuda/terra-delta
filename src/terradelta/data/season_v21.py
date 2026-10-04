"""Bounded seasonal augmentation; no changes to the v2 inference architecture."""
from __future__ import annotations

import math
import operator

import numpy as np
import torch
import torch.nn.functional as F

from .dataset_v2 import MEAN, STD, V2Transform, _photo, _tensor


# In RGB [0, 1], including the optional v2 photometry in SeasonTransform.
MAX_RGB_DELTA = .24


def _bounded(name, value, maximum):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [0, {maximum}]")
    return value


def _seed(seed):
    try:
        return operator.index(seed) % 2**32
    except TypeError as exc:
        raise ValueError("seed must be an integer") from exc


def _normalization(image):
    return (image.new_tensor(MEAN).view(1, 3, 1, 1),
            image.new_tensor(STD).view(1, 3, 1, 1))


def _appearance(rgb, rng, strength):
    """One RGB tile, with broad illumination fields and a symmetric mild blur.

    Randomness is local NumPy state; all pixel operations stay on rgb.device.
    The soft excess-green weight is an appearance heuristic, never a class mask.
    """
    h, w = rgb.shape[-2:]
    gamma = rng.uniform(.80, 1.25)
    contrast = rng.uniform(.85, 1.15)
    brightness = rng.uniform(-.055, .055)
    tint = rgb.new_tensor(rng.uniform([-.06, -.12, -.08], [.12, .07, .06])).view(1, 3, 1, 1)
    shadow = rng.uniform(.03, .18)
    haze = rng.uniform(0, .10)
    local_brightness = rng.uniform(-.08, .08)
    local_contrast = rng.uniform(-.12, .12)
    sigma = rng.uniform(.35, .85)
    blur_mix = rng.uniform(0, .30)

    # Smooth fields computed at the original pixel coordinates; no resampling.
    y = torch.linspace(0, 1, h, device=rgb.device, dtype=torch.float32).view(1, 1, h, 1)
    x = torch.linspace(0, 1, w, device=rgb.device, dtype=torch.float32).view(1, 1, 1, w)
    cx, cy = rng.uniform(0, 1, 2)
    radius = rng.uniform(.25, .65)
    field = torch.exp(-((x - cx).square() + (y - cy).square()) / (2 * radius**2))
    angle = rng.uniform(0, 2 * np.pi)
    shade = torch.sigmoid(5 * ((x - .5) * math.cos(angle) + (y - .5) * math.sin(angle)))
    green = ((rgb[:, 1:2] - (rgb[:, :1] + rgb[:, 2:3]) / 2) / .20).clamp(0, 1)
    out = rgb.clamp_min(1e-6).pow(1 + strength * (gamma - 1))
    out = (out - .5) * (1 + strength * (contrast - 1)) + .5 + strength * brightness
    out = out + strength * green * tint
    out = (out - .5) * (1 + strength * local_contrast * field) + .5
    out = out + strength * local_brightness * field
    out = out * (1 - strength * shadow * shade)
    out = out * (1 - strength * haze) + strength * haze * .92

    axis = torch.arange(-1, 2, device=rgb.device, dtype=torch.float32)
    kernel = torch.exp(-(axis[:, None].square() + axis[None, :].square()) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).expand(3, 1, 3, 3)
    blurred = F.conv2d(F.pad(out, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
    out = out + strength * blur_mix * (blurred - out)
    limit = MAX_RGB_DELTA * strength
    return torch.maximum(torch.minimum(out, rgb + limit), rgb - limit).clamp(0, 1)


def photometric_view(image: torch.Tensor, seed: int, *, strength: float = 1.0) -> torch.Tensor:
    """Independent PRE/POST appearance view of normalized B6HW (ImageNet RGB).

    No flips, rotation, crop, registration or mask operations. Returns finite
    float32 on the input device, bounded to normalized RGB [0, 1], without
    mutating the input or global random state. Each batch member and date gets
    independent parameters. The seed reproduces a view on the same device;
    strength=0 preserves in-range input values exactly after float32 conversion.
    Autograd is retained, although training data normally needs no gradients.
    """
    strength = _bounded("strength", strength, 1.0)
    seed = _seed(seed)
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 6:
        raise ValueError("image must be a normalized B6HW torch tensor")
    if any(size == 0 for size in image.shape) or not image.is_floating_point():
        raise ValueError("image must have nonempty dimensions and floating dtype")
    image = image.to(dtype=torch.float32)
    if not torch.isfinite(image).all():
        raise ValueError("image must contain only finite float32 values")
    mean, std = _normalization(image)
    means, stds = mean.repeat(1, 2, 1, 1), std.repeat(1, 2, 1, 1)
    if strength == 0:
        return torch.maximum(torch.minimum(image, (1 - means) / stds), -means / stds).clone()
    rng = np.random.default_rng(seed)
    rgb = (image * stds + means).clamp(0, 1)
    views = [torch.cat([_appearance(tile[None, :3], rng, strength),
                        _appearance(tile[None, 3:], rng, strength)], dim=1)
             for tile in rgb]
    return (torch.cat(views, dim=0) - means) / stds


def photometric_view_numpy(pre, post, seed, *, strength=1.0):
    """CPU uint8 RGB pair adapter to the same tensor appearance implementation."""
    _validate_pair(pre, post)
    image = torch.cat([_tensor(pre), _tensor(post)])[None]
    view = photometric_view(image, seed, strength=strength)[0]
    mean, std = _normalization(view)
    outputs = []
    for tile in (view[:3], view[3:]):
        rgb = (tile * std[0] + mean[0]).clamp(0, 1)
        outputs.append(np.ascontiguousarray((rgb * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()))
    return tuple(outputs)


def _validate_pair(pre, post):
    if (not isinstance(pre, np.ndarray) or not isinstance(post, np.ndarray)
            or pre.ndim != 3 or pre.shape[-1] != 3 or pre.shape != post.shape
            or min(pre.shape[:2]) == 0 or pre.dtype != np.uint8 or post.dtype != np.uint8):
        raise ValueError("pre/post must be matching nonempty HWC uint8 RGB arrays")


class SeasonTransform:
    """V2 geometry, mild PRE-only registration, independent bounded seasons.

    Masks and validity undergo only the original v2 shared geometry. Strength
    is in [0, 1]; appearance changes are at most 0.24 * strength per RGB channel
    relative to the geometry-transformed input (plus uint8 rounding). Registration
    defaults are milder than v2 and cannot exceed its translation/rotation/scale
    bounds. photometric=False disables all appearance changes.
    """

    def __init__(self, *, strength=1.0, relative_probability=.55,
                 max_translation=3.0, max_rotation=.8, max_scale=.01, photometric=True):
        self.strength = _bounded("strength", strength, 1.0)
        self.geometry = V2Transform(
            relative_probability=_bounded("relative_probability", relative_probability, 1.0),
            max_translation=_bounded("max_translation", max_translation, 6.0),
            max_rotation=_bounded("max_rotation", max_rotation, 1.5),
            max_scale=_bounded("max_scale", max_scale, .02),
            photometric=False,
        )
        if not isinstance(photometric, (bool, np.bool_)):
            raise ValueError("photometric must be a boolean")
        self.photometric = bool(photometric)

    def __call__(self, pre, post, masks, valid, seed):
        _validate_pair(pre, post)
        if masks.shape != (2, *pre.shape[:2]) or valid.shape != pre.shape[:2]:
            raise ValueError("masks must be 2HW and valid must be HW matching the images")
        seed = _seed(seed)
        pre, post, masks, valid = self.geometry(pre, post, masks, valid, seed)
        if self.photometric and self.strength:
            # Separate appearance stream keeps common geometry identical to v2.
            rng = np.random.default_rng(np.random.SeedSequence([seed, 21]))
            originals = pre, post
            bases = []
            for tile in originals:
                photo = _photo(tile, rng, strength=.4 * self.strength)
                bases.append(np.clip(tile.astype(np.float32) + self.strength *
                                     (photo.astype(np.float32) - tile), 0, 255).round().astype(np.uint8))
            views = photometric_view_numpy(*bases, seed=int(rng.integers(0, 2**32)), strength=self.strength)
            limit = MAX_RGB_DELTA * self.strength * 255
            pre, post = tuple(np.ascontiguousarray(np.clip(view.astype(np.float32),
                             original.astype(np.float32) - limit,
                             original.astype(np.float32) + limit).round().astype(np.uint8))
                             for original, view in zip(originals, views))
        return pre, post, masks, valid
