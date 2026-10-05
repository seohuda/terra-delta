"""Unit tests for v2.3.2 local alignment residual evidence (tiny synthetic arrays only)."""
import json
import math
import numpy as np
import pytest

from terradelta.inference.alignment_cache import CACHE_VERSION, AlignmentCacheKey, AlignmentFeatureCache
from terradelta.inference.local_alignment import (
    ALIGNMENT_VERSION,
    EPS,
    V232_ALIGNMENT_FEATURES,
    LocalAlignmentConfig,
    LocalAlignmentFeatureExtractor,
    mask_bbox,
    mask_fingerprint,
    scale_offsets,
)


def textured(c=4, h=32, w=32, seed=0):
    return np.random.default_rng(seed).normal(size=(c, h, w)).astype(np.float32)


def square_mask(y0, x0, y1, x1, size=32):
    m = np.zeros((size, size), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def ext(**kw):
    base = dict(
        image_size=32,
        bbox_padding=4,
        ring_radius=2,
        max_shift_image_px=3,
        feature_scales=(0,),
        min_overlap_fraction=0.75,
    )
    base.update(kw)
    return LocalAlignmentFeatureExtractor(LocalAlignmentConfig(**base))


def shifted(arr, dy, dx):
    """post(y+dy, x+dx) == pre(y, x)."""
    return np.roll(arr, shift=(dy, dx), axis=(1, 2))


def test_identical_maps_give_zero_residual_semantics():
    """Item 3: identical non-zero maps must report ratio=1.0, reduction=0.0."""
    f = textured()
    r = ext().extract_scale(square_mask(10, 10, 18, 18), f, f.copy(), 0)
    assert r.valid
    assert r.residual_before == pytest.approx(0.0, abs=1e-6)
    assert r.residual_after == pytest.approx(0.0, abs=1e-6)
    assert r.residual_ratio == 1.0
    assert r.residual_reduction == 0.0
    assert (r.best_dx, r.best_dy) == (0, 0)
    assert r.cosine_before == pytest.approx(1.0, abs=1e-5)
    assert math.isfinite(r.residual_ratio)
    assert math.isfinite(r.residual_reduction)


def test_identical_zero_maps():
    """Item 3: identical all-zero maps must not produce NaN/inf or false reduction."""
    z = np.zeros((4, 32, 32), dtype=np.float32)
    r = ext().extract_scale(square_mask(10, 10, 18, 18), z, z.copy(), 0)
    assert r.valid
    assert r.residual_before == 0.0
    assert r.residual_after == 0.0
    assert r.residual_ratio == 1.0
    assert r.residual_reduction == 0.0
    assert (r.best_dx, r.best_dy) == (0, 0)
    assert math.isfinite(r.residual_ratio)
    assert math.isfinite(r.residual_reduction)
    assert math.isfinite(r.cosine_before)


def test_nearly_identical_maps():
    """Item 3: maps differing by <= EPS must trigger the safe zero-residual fallback."""
    pre = textured()
    post = pre + np.float32(EPS / 2.0)
    r = ext().extract_scale(square_mask(10, 10, 18, 18), pre, post, 0)
    assert r.valid
    assert r.residual_ratio == 1.0
    assert r.residual_reduction == 0.0
    assert (r.best_dx, r.best_dy) == (0, 0)


@pytest.mark.parametrize("dy,dx", [(0, 1), (1, 0), (-2, 1), (2, -2)])
def test_synthetic_shift_recovered(dy, dx):
    pre = textured()
    post = shifted(pre, dy, dx)
    r = ext(max_shift_image_px=3).extract_scale(square_mask(12, 12, 20, 20), pre, post, 0)
    assert (r.best_dy, r.best_dx) == (dy, dx)
    assert r.residual_after < 1e-5 < r.residual_before
    assert r.residual_reduction == pytest.approx(1.0, abs=1e-4)
    assert r.cosine_gain > 0
    assert r.best_shift_distance == pytest.approx(np.hypot(dy, dx))


def test_real_difference_remains_after_alignment():
    pre = textured(seed=1)
    post = textured(seed=2)  # unrelated content: no translation explains it
    r = ext().extract_scale(square_mask(10, 10, 20, 20), pre, post, 0)
    assert r.valid
    assert r.residual_ratio > 0.8
    assert r.residual_after > 0.5


def test_shift_search_respects_image_budget():
    """Item 1: shifts exceeding image-space budget are not evaluated."""
    pre = textured()
    post = shifted(pre, 0, 5)  # beyond max_shift_image_px=2
    r = ext(max_shift_image_px=2).extract_scale(square_mask(12, 12, 20, 20), pre, post, 0)
    assert abs(r.best_dx) <= 2 and abs(r.best_dy) <= 2
    assert r.residual_after > 1e-3


def test_scale_offsets_image_space_budget():
    """Item 1: derive integer feature-map shifts and verify coarse scale fallback."""
    # Scale 2 at 64x64 on 256 image -> stride 4.0 px/cell
    offsets_64 = scale_offsets(max_shift_image_px=12, image_size=256, fh=64, fw=64)
    assert (0, 0) in offsets_64
    assert (3, 0) in offsets_64
    assert (0, 3) in offsets_64
    # (3, 3) would be sqrt(12^2 + 12^2) = 16.97 > 12 -> excluded
    assert (3, 3) not in offsets_64
    assert (-3, -3) not in offsets_64

    # Coarse scale: 16x16 on 256 image -> stride 16.0 px/cell > 12 -> zero-only
    offsets_16 = scale_offsets(max_shift_image_px=12, image_size=256, fh=16, fw=16)
    assert offsets_16 == [(0, 0)]

    # Zero budget: returns [(0, 0)]
    assert scale_offsets(max_shift_image_px=0, image_size=256, fh=64, fw=64) == [(0, 0)]


def test_adversarial_border_candidate_left_edge():
    """Item 2: candidate touching left edge with mismatch at border cannot win by dropping pixels."""
    # Construct PRE and POST such that interior is identical, but col 0 has large mismatch
    pre = np.ones((4, 32, 32), dtype=np.float32)
    post = pre.copy()
    post[:, :, 0] = 50.0  # Large mismatch strictly at the left edge
    # Mask touches left edge (x from 0 to 8)
    mask = square_mask(10, 0, 18, 8)
    r = ext(max_shift_image_px=2).extract_scale(mask, pre, post, 0)
    # The negative x shift (which would push col 0 out of bounds) MUST NOT win over (0, 0)
    assert r.valid
    assert r.best_dx >= 0, f"Out-of-bounds negative shift dx={r.best_dx} won unfairly!"
    assert 0.0 < r.valid_fraction <= 1.0


def test_adversarial_border_candidate_top_edge():
    """Item 2: candidate touching top edge with mismatch at top row cannot win by dropping pixels."""
    pre = np.ones((4, 32, 32), dtype=np.float32)
    post = pre.copy()
    post[:, 0, :] = 50.0  # Large mismatch strictly at the top row
    mask = square_mask(0, 10, 8, 18)
    r = ext(max_shift_image_px=2).extract_scale(mask, pre, post, 0)
    assert r.valid
    assert r.best_dy >= 0, f"Out-of-bounds negative shift dy={r.best_dy} won unfairly!"
    assert 0.0 < r.valid_fraction <= 1.0


def test_very_small_object():
    pre = textured()
    r = ext(min_valid_pixels=1).extract_scale(square_mask(15, 15, 16, 16), pre, pre.copy(), 0)
    assert r.valid and r.residual_after == pytest.approx(0.0, abs=1e-6)


def test_empty_mask_and_invalid_inputs():
    f = textured()
    r = ext().extract_scale(np.zeros((32, 32), dtype=bool), f, f, 0)
    assert not r.valid and r.valid_fraction == 0.0
    with pytest.raises(ValueError, match="share shape"):
        ext().extract_scale(square_mask(1, 1, 4, 4), f, f[:, :16], 0)
    bad = f.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ext().extract_scale(square_mask(1, 1, 4, 4), bad, f, 0)
    with pytest.raises(ValueError, match="2-D"):
        ext().extract_scale(np.zeros((1, 32, 32), dtype=bool), f, f, 0)


def test_too_few_valid_pixels_is_invalid():
    f = textured()
    r = ext(min_valid_pixels=10_000).extract_scale(square_mask(10, 10, 14, 14), f, f, 0)
    assert not r.valid


def test_deterministic_output():
    pre, post = textured(seed=3), textured(seed=4)
    mask = square_mask(8, 8, 16, 16)
    a = ext().extract(mask, {0: pre}, {0: post})
    b = ext().extract(mask, {0: pre}, {0: post})
    assert a == b
    assert tuple(a) == V232_ALIGNMENT_FEATURES


def test_multiscale_aggregation_and_resolution_mapping():
    pre32, pre16 = textured(h=32, w=32, seed=5), textured(h=16, w=16, seed=6)
    e = ext(feature_scales=(0, 1))
    feats = e.extract(square_mask(8, 8, 24, 24), {0: pre32, 1: pre16}, {0: pre32.copy(), 1: pre16.copy()})
    assert feats["align_num_valid_scales"] == 2.0
    assert feats["align_residual_after"] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="Missing encoder feature scale"):
        e.extract(square_mask(8, 8, 24, 24), {0: pre32}, {0: pre32})


def test_shift_reported_in_image_pixels_at_coarse_scale():
    pre = textured(h=16, w=16, seed=7)
    post = shifted(pre, 0, 1)  # 1 feature px == 2 image px
    feats = ext(feature_scales=(0,), bbox_padding=8, ring_radius=4, max_shift_image_px=4).extract(
        square_mask(8, 8, 24, 24), {0: pre}, {0: post}
    )
    assert feats["align_best_dx"] == pytest.approx(2.0)


def test_all_invalid_gives_zero_features():
    f = textured()
    feats = ext().extract(np.zeros((32, 32), dtype=bool), {0: f}, {0: f})
    assert all(v == 0.0 for v in feats.values())


def test_config_validation_and_metadata():
    assert LocalAlignmentConfig().enabled is False
    assert LocalAlignmentConfig.from_mapping(None).enabled is False
    with pytest.raises(ValueError, match="Unknown"):
        LocalAlignmentConfig.from_mapping({"bogus": 1})
    with pytest.raises(ValueError, match="metric"):
        LocalAlignmentConfig(metric="l3")
    with pytest.raises(ValueError, match="max_shift_image_px"):
        LocalAlignmentConfig(max_shift_image_px=-1)
    with pytest.raises(ValueError, match="min_overlap_fraction"):
        LocalAlignmentConfig(min_overlap_fraction=0.0)
    with pytest.raises(ValueError, match="min_overlap_fraction"):
        LocalAlignmentConfig(min_overlap_fraction=1.5)
    with pytest.raises(ValueError, match="unique"):
        LocalAlignmentConfig(feature_scales=(2, 2))

    cfg = LocalAlignmentConfig.from_mapping({"enabled": True, "feature_scales": [1, 2]})
    assert cfg.feature_scales == (1, 2)
    meta = cfg.metadata()
    assert meta["algorithm_version"] == ALIGNMENT_VERSION
    assert meta["feature_schema"] == list(V232_ALIGNMENT_FEATURES)
    assert cfg.fingerprint() == LocalAlignmentConfig(feature_scales=(1, 2)).fingerprint()
    assert cfg.fingerprint() != LocalAlignmentConfig(feature_scales=(1, 3)).fingerprint()


def test_mask_helpers():
    m = square_mask(2, 3, 5, 9)
    assert mask_bbox(m) == (2, 3, 5, 9)
    assert mask_bbox(np.zeros((4, 4), dtype=bool)) is None
    assert mask_fingerprint(m) == mask_fingerprint(m.copy())
    assert mask_fingerprint(m) != mask_fingerprint(square_mask(2, 3, 5, 10))


def test_cache_roundtrip_and_fingerprints(tmp_path):
    """Item 5: cache requires model and pair fingerprints, validates finite values."""
    cfg = LocalAlignmentConfig(image_size=32, feature_scales=(0,), bbox_padding=4, ring_radius=2)
    e = LocalAlignmentFeatureExtractor(cfg)
    mask = square_mask(8, 8, 16, 16)
    bbox = mask_bbox(mask)
    mfp = "model-sha-123456"
    pfp = "pair-sha-abcdef"
    assert CACHE_VERSION == 2

    # Missing fingerprint raises ValueError
    with pytest.raises(ValueError, match="model_fingerprint"):
        AlignmentCacheKey.build("p1", "new_building", 0, bbox, mask_fingerprint(mask), cfg, "", pfp)
    with pytest.raises(ValueError, match="pair_fingerprint"):
        AlignmentCacheKey.build("p1", "new_building", 0, bbox, mask_fingerprint(mask), cfg, mfp, "")

    key = AlignmentCacheKey.build("p1", "new_building", 0, bbox, mask_fingerprint(mask), cfg, mfp, pfp)
    cache = AlignmentFeatureCache()
    pre, post = textured(seed=8), textured(seed=9)
    first = e.extract(mask, {0: pre}, {0: post}, cache=cache, cache_key=key)
    assert len(cache) == 1

    # Cache hit returns identical result
    assert e.extract(mask, {0: pre * 0}, {0: post * 0}, cache=cache, cache_key=key) == first

    # Persistence to file (version 2)
    path = tmp_path / "c.json"
    cache.save(path)
    loaded = AlignmentFeatureCache.load(path)
    assert loaded.get(key) == first

    # Old version 1 cache file rejected
    old_data = {"version": 1, "entries": {}}
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old_data), encoding="utf-8")
    with pytest.raises(ValueError, match="version 2"):
        AlignmentFeatureCache.load(old_path)

    # Incomplete features rejected
    with pytest.raises(ValueError, match="incomplete"):
        cache.put(key, {"align_best_dx": 0.0})

    # Non-finite values rejected
    bad_features = dict(first)
    bad_features["align_residual_before"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        cache.put(key, bad_features)
