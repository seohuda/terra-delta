"""Local feature-space alignment residual evidence (v2.3.2 experimental, diagnostic only).

For one candidate component, compare frozen Siamese encoder features of PRE and POST inside a
bounded ROI, search small integer translations of POST relative to PRE and report how much of the
apparent feature change survives the best translation:

* change mostly disappears after a tiny shift -> likely registration / parallax artifact
* change remains after the best shift          -> more likely real temporal change

Convention: ``best_dx``/``best_dy`` mean POST is sampled at ``(y + dy, x + dx)`` for a PRE pixel
at ``(y, x)``. Shifts are in *feature-map* pixels in :class:`LocalAlignmentResult` and are
converted to image pixels in the aggregated feature dictionary.

This module never touches the segmentation path, the encoder weights or FlowAlign. Runtime
dependencies: stdlib + NumPy.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

ALIGNMENT_VERSION = "v232-align-1"
EPS = 1e-8
METRICS = ("l1", "l2")

# Deterministic aggregated schema. Kept separate from the frozen V2.3.1 ablation schemas.
V232_ALIGNMENT_FEATURES: tuple[str, ...] = (
    "align_best_dx",
    "align_best_dy",
    "align_shift_distance",
    "align_residual_before",
    "align_residual_after",
    "align_residual_ratio",
    "align_residual_reduction",
    "align_cosine_before",
    "align_cosine_after",
    "align_cosine_gain",
    "align_norm_l1_before",
    "align_norm_l1_after",
    "align_norm_l2_before",
    "align_norm_l2_after",
    "align_interior_residual_after",
    "align_ring_residual_after",
    "align_valid_fraction",
    "align_num_valid_scales",
)


@dataclass(frozen=True)
class LocalAlignmentConfig:
    """Configuration; ``enabled`` defaults to False so the V2.3.1 path is untouched."""

    enabled: bool = False
    max_shift: int = 3  # feature-map pixels, search window is [-max_shift, max_shift]^2
    bbox_padding: int = 8  # image pixels added around the candidate bbox
    feature_scales: tuple[int, ...] = (2, 3)  # indices into the encoder feature pyramid
    metric: str = "l2"  # residual used to pick the best shift: "l1" or "l2"
    ring_radius: int = 5  # image pixels of surrounding context included in the comparison
    image_size: int = 256
    min_valid_pixels: int = 4

    def __post_init__(self) -> None:
        for name in ("max_shift", "bbox_padding", "ring_radius", "min_valid_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"alignment_residual.{name} must be a non-negative integer")
        if self.min_valid_pixels < 1:
            raise ValueError("alignment_residual.min_valid_pixels must be >= 1")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int) or self.image_size < 1:
            raise ValueError("alignment_residual.image_size must be a positive integer")
        if not isinstance(self.enabled, bool):
            raise ValueError("alignment_residual.enabled must be a boolean")
        if self.metric not in METRICS:
            raise ValueError(f"alignment_residual.metric must be one of {METRICS}")
        scales = tuple(self.feature_scales)
        if not scales or len(set(scales)) != len(scales):
            raise ValueError("alignment_residual.feature_scales must be non-empty and unique")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in scales):
            raise ValueError("alignment_residual.feature_scales must be non-negative integers")
        object.__setattr__(self, "feature_scales", scales)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> LocalAlignmentConfig:
        """Parse an ``experimental.alignment_residual`` mapping; ``None`` means defaults (off)."""
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("alignment_residual config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown alignment_residual keys: {sorted(unknown)}")
        kwargs = dict(data)
        if "feature_scales" in kwargs:
            kwargs["feature_scales"] = tuple(kwargs["feature_scales"])
        return cls(**kwargs)

    def fingerprint(self) -> str:
        """Stable hash of everything that changes feature values (cache invalidation key)."""
        payload = asdict(self)
        payload.pop("enabled")
        payload["version"] = ALIGNMENT_VERSION
        text = json.dumps(payload, sort_keys=True, default=list)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LocalAlignmentResult:
    """Per-scale alignment evidence. Shifts are in feature-map pixels."""

    scale: int
    valid: bool
    best_dx: int = 0
    best_dy: int = 0
    best_shift_distance: float = 0.0
    residual_before: float = 0.0
    residual_after: float = 0.0
    residual_ratio: float = 0.0
    residual_reduction: float = 0.0
    cosine_before: float = 0.0
    cosine_after: float = 0.0
    cosine_gain: float = 0.0
    normalized_l1_before: float = 0.0
    normalized_l1_after: float = 0.0
    normalized_l2_before: float = 0.0
    normalized_l2_after: float = 0.0
    interior_residual_after: float = 0.0
    ring_residual_after: float = 0.0
    valid_fraction: float = 0.0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(y0, x0, y1, x1)`` (exclusive upper bounds) of a boolean mask, or None if empty."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def mask_fingerprint(mask: np.ndarray) -> str:
    """Content hash of a candidate mask (for cache keys)."""
    m = np.ascontiguousarray(mask.astype(bool))
    h = hashlib.sha256()
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(np.packbits(m).tobytes())
    return h.hexdigest()[:16]


def _to_feature_mask(mask: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """Max-pool an image-resolution mask to feature resolution (any covered pixel marks a cell)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    out = np.zeros((fh, fw), dtype=bool)
    out[ys * fh // h, xs * fw // w] = True
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square binary dilation by shifting; deterministic and dependency free."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    padded[radius : radius + h, radius : radius + w] = mask
    out = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def _shift_offsets(max_shift: int) -> list[tuple[int, int]]:
    """Search offsets ordered by distance then (dy, dx): ties resolve to the smaller shift."""
    offsets = [(dy, dx) for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)]
    return sorted(offsets, key=lambda o: (o[0] ** 2 + o[1] ** 2, o[0], o[1]))


@dataclass(frozen=True)
class _ShiftStats:
    residual: float
    cosine: float
    norm_l1: float
    norm_l2: float
    interior: float
    ring: float
    valid_fraction: float
    n_valid: int


def _stats_at_shift(
    pre: np.ndarray,
    post: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    interior: np.ndarray,
    dy: int,
    dx: int,
    metric: str,
) -> _ShiftStats | None:
    """Residual statistics comparing PRE at (ys, xs) with POST at (ys + dy, xs + dx)."""
    _, fh, fw = pre.shape
    qy, qx = ys + dy, xs + dx
    ok = (qy >= 0) & (qy < fh) & (qx >= 0) & (qx < fw)
    n_valid = int(ok.sum())
    if n_valid == 0:
        return None
    a = pre[:, ys[ok], xs[ok]].astype(np.float64).T  # (N, C)
    b = post[:, qy[ok], qx[ok]].astype(np.float64).T
    diff = a - b
    l1 = np.abs(diff).sum(axis=1)
    l2 = np.sqrt((diff**2).sum(axis=1))
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = (a * b).sum(axis=1) / np.maximum(na * nb, EPS)
    per_pixel = l1 if metric == "l1" else l2
    inner = interior[ok]
    ring = ~inner
    scale_l1 = float(((np.abs(a).sum(axis=1) + np.abs(b).sum(axis=1)) / 2).mean())
    scale_l2 = float(((na + nb) / 2).mean())
    return _ShiftStats(
        residual=float(per_pixel.mean()),
        cosine=float(cos.mean()),
        norm_l1=float(l1.mean() / max(scale_l1, EPS)),
        norm_l2=float(l2.mean() / max(scale_l2, EPS)),
        interior=float(per_pixel[inner].mean()) if inner.any() else 0.0,
        ring=float(per_pixel[ring].mean()) if ring.any() else 0.0,
        valid_fraction=n_valid / len(ys),
        n_valid=n_valid,
    )


class LocalAlignmentFeatureExtractor:
    """Compute alignment evidence for candidate masks from cached encoder feature maps."""

    def __init__(self, config: LocalAlignmentConfig):
        self.config = config

    def roi_region(self, mask: np.ndarray, fh: int, fw: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Feature-resolution region (interior flag + membership) inside the padded bbox ROI.

        Returns ``(region, interior)`` boolean maps, or None when the mask is empty. Only
        interior cells and a thin surrounding ring are compared so empty background cannot
        dominate the metric.
        """
        cfg = self.config
        bbox = mask_bbox(mask)
        if bbox is None:
            return None
        h, w = mask.shape
        y0, x0, y1, x1 = bbox
        pad = cfg.bbox_padding
        ry0, rx0 = max(y0 - pad, 0), max(x0 - pad, 0)
        ry1, rx1 = min(y1 + pad, h), min(x1 + pad, w)
        roi = np.zeros((fh, fw), dtype=bool)
        roi[ry0 * fh // h : -(-ry1 * fh // h), rx0 * fw // w : -(-rx1 * fw // w)] = True
        interior = _to_feature_mask(mask, fh, fw)
        ring_r = max(1, math.ceil(cfg.ring_radius * fh / h)) if cfg.ring_radius > 0 else 0
        region = (_dilate(interior, ring_r) if ring_r else interior) & roi
        return region, interior & region

    def extract_scale(
        self, mask: np.ndarray, pre: np.ndarray, post: np.ndarray, scale: int
    ) -> LocalAlignmentResult:
        """Alignment evidence for one encoder scale; ``pre``/``post`` are ``(C, H, W)`` arrays."""
        if pre.ndim != 3 or pre.shape != post.shape:
            raise ValueError("pre/post feature maps must share shape (C, H, W)")
        for arr in (pre, post):
            if not np.isfinite(arr).all():
                raise ValueError("feature maps contain NaN/inf")
        if mask.ndim != 2:
            raise ValueError("mask must be 2-D")
        _, fh, fw = pre.shape
        roi = self.roi_region(mask.astype(bool), fh, fw)
        if roi is None:
            return LocalAlignmentResult(scale=scale, valid=False)
        region, interior = roi
        ys, xs = np.nonzero(region)
        inner = interior[ys, xs]
        base = _stats_at_shift(pre, post, ys, xs, inner, 0, 0, self.config.metric)
        if base is None or base.n_valid < self.config.min_valid_pixels:
            return LocalAlignmentResult(scale=scale, valid=False)

        best, best_off = base, (0, 0)
        for dy, dx in _shift_offsets(self.config.max_shift):
            if (dy, dx) == (0, 0):
                continue
            stats = _stats_at_shift(pre, post, ys, xs, inner, dy, dx, self.config.metric)
            if stats is None or stats.n_valid < self.config.min_valid_pixels:
                continue
            if stats.residual < best.residual - 1e-12:  # strict: earlier (smaller) shift wins ties
                best, best_off = stats, (dy, dx)
        before, after = base.residual, best.residual
        return LocalAlignmentResult(
            scale=scale,
            valid=True,
            best_dx=best_off[1],
            best_dy=best_off[0],
            best_shift_distance=math.hypot(*best_off),
            residual_before=before,
            residual_after=after,
            residual_ratio=after / max(before, EPS),
            residual_reduction=(before - after) / max(before, EPS),
            cosine_before=base.cosine,
            cosine_after=best.cosine,
            cosine_gain=best.cosine - base.cosine,
            normalized_l1_before=base.norm_l1,
            normalized_l1_after=best.norm_l1,
            normalized_l2_before=base.norm_l2,
            normalized_l2_after=best.norm_l2,
            interior_residual_after=best.interior,
            ring_residual_after=best.ring,
            valid_fraction=best.valid_fraction,
        )

    def aggregate(
        self, results: Sequence[LocalAlignmentResult], feature_shapes: Mapping[int, tuple[int, int]]
    ) -> dict[str, float]:
        """Mean over valid scales in config order; shifts converted to image pixels.

        If no scale is valid every feature is 0.0 (``align_valid_fraction`` = 0 flags it).
        """
        valid = [r for r in results if r.valid]
        out = {name: 0.0 for name in V232_ALIGNMENT_FEATURES}
        if not valid:
            return out
        size = self.config.image_size

        def mean(fn) -> float:
            return float(np.mean([fn(r) for r in valid]))

        def px(r: LocalAlignmentResult) -> float:
            return size / feature_shapes[r.scale][0]

        out.update(
            align_best_dx=mean(lambda r: r.best_dx * px(r)),
            align_best_dy=mean(lambda r: r.best_dy * px(r)),
            align_shift_distance=mean(lambda r: r.best_shift_distance * px(r)),
            align_residual_before=mean(lambda r: r.residual_before),
            align_residual_after=mean(lambda r: r.residual_after),
            align_residual_ratio=mean(lambda r: r.residual_ratio),
            align_residual_reduction=mean(lambda r: r.residual_reduction),
            align_cosine_before=mean(lambda r: r.cosine_before),
            align_cosine_after=mean(lambda r: r.cosine_after),
            align_cosine_gain=mean(lambda r: r.cosine_gain),
            align_norm_l1_before=mean(lambda r: r.normalized_l1_before),
            align_norm_l1_after=mean(lambda r: r.normalized_l1_after),
            align_norm_l2_before=mean(lambda r: r.normalized_l2_before),
            align_norm_l2_after=mean(lambda r: r.normalized_l2_after),
            align_interior_residual_after=mean(lambda r: r.interior_residual_after),
            align_ring_residual_after=mean(lambda r: r.ring_residual_after),
            align_valid_fraction=mean(lambda r: r.valid_fraction),
            align_num_valid_scales=float(len(valid)),
        )
        return out

    def extract(
        self,
        mask: np.ndarray,
        pre_feats: Mapping[int, np.ndarray],
        post_feats: Mapping[int, np.ndarray],
        cache: Any = None,
        cache_key: Any = None,
    ) -> dict[str, float]:
        """Aggregated alignment features for one candidate mask, optionally cache-backed.

        ``pre_feats``/``post_feats`` map encoder scale index -> ``(C, H, W)`` array and must
        contain every configured scale. ``cache``/``cache_key`` follow
        :class:`terradelta.inference.alignment_cache.AlignmentFeatureCache`.
        """
        if cache is not None and cache_key is not None:
            hit = cache.get(cache_key)
            if hit is not None:
                return hit
        results, shapes = [], {}
        for scale in self.config.feature_scales:
            if scale not in pre_feats or scale not in post_feats:
                raise ValueError(f"Missing encoder feature scale {scale}")
            pre, post = np.asarray(pre_feats[scale]), np.asarray(post_feats[scale])
            shapes[scale] = (pre.shape[1], pre.shape[2])
            results.append(self.extract_scale(mask, pre, post, scale))
        features = self.aggregate(results, shapes)
        if cache is not None and cache_key is not None:
            cache.put(cache_key, features)
        return features
