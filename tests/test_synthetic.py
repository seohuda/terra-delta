"""Substantive mock-only tests: no files downloaded, training, or optimizer steps."""

import json

import cv2
import numpy as np
import pytest

from terradelta.data.forest_labels import ForestLossRefiner, refine_forest_loss
from terradelta.data.negatives import (
    EXTRA_NEGATIVE_TYPES,
    NEGATIVE_TYPES,
    NegativePairGenerator,
    generate_negative_pair,
)
from terradelta.data.synthetic_building import (
    SyntheticBuildingGenerator,
    edge_feather,
    generate_synthetic_building_pair,
    patch_fill,
    telea_fill,
    texture_copy_fill,
)


@pytest.fixture
def building_scene():
    rng = np.random.default_rng(88)
    image = np.clip(np.array([60, 125, 65]) + rng.integers(-18, 19, (64, 80, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((64, 80), dtype=np.uint8)
    mask[22:38, 30:46] = 255
    image[mask != 0] = [215, 195, 185]
    return image, mask


def assert_pair_arrays(pair, shape):
    for key in ("pre", "post"):
        assert pair[key].shape == (*shape, 3)
        assert pair[key].dtype == np.uint8
    for key in ("new_building", "tree_removal"):
        assert pair[key].shape == shape
        assert pair[key].dtype == np.uint8
        assert set(np.unique(pair[key])) <= {0, 1}
    assert not np.shares_memory(pair["pre"], pair["post"])
    json.dumps(pair["metadata"], allow_nan=False)


@pytest.mark.parametrize("strategy", ["telea", "texture_copy", "patch_fill"])
@pytest.mark.parametrize("bool_mask", [False, True])
def test_building_removes_roof_preserves_gt_and_outside_halo(building_scene, strategy, bool_mask):
    post, footprint = building_scene
    original = post.copy()
    footprint = footprint.astype(bool) if bool_mask else footprint
    generator = SyntheticBuildingGenerator(strategy=strategy, seed=7, feather_radius=3)
    pair = generator(post, footprint)
    assert_pair_arrays(pair, footprint.shape)
    assert np.array_equal(pair["post"], original)
    assert np.array_equal(post, original)
    assert np.array_equal(pair["new_building"], footprint != 0)
    assert not pair["tree_removal"].any()
    # All interior roof pixels replaced with plausible surrounding terrain, not black.
    assert pair["pre"][footprint != 0].mean() < 140
    assert pair["pre"][footprint != 0].min() > 0
    assert (pair["pre"][footprint != 0].astype(int) - original[footprint != 0]).mean() < -40
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    halo = cv2.dilate((footprint != 0).astype(np.uint8), kernel).astype(bool)
    assert np.array_equal(pair["pre"][~halo], original[~halo])
    assert pair["metadata"]["edit_pixels"] == int(halo.sum())
    repeat = generator(post, footprint)
    assert np.array_equal(pair["pre"], repeat["pre"])
    assert pair["metadata"] == repeat["metadata"]


def test_feather_fully_replaces_footprint_blends_outer_edge_only():
    original = np.full((15, 15, 3), 100, dtype=np.uint8)
    filled = np.full_like(original, 20)
    mask = np.zeros((15, 15), dtype=bool)
    mask[5:10, 5:10] = True
    feathered = edge_feather(original, filled, mask, radius=2)
    assert np.all(feathered[mask] == 20)
    assert np.all((feathered[4, 7] > 20) & (feathered[4, 7] < 100))
    assert np.all(feathered[0] == 100)
    hard = edge_feather(original, filled, mask, radius=0)
    assert np.all(hard[~mask] == 100)
    assert np.all(hard[mask] == 20)


@pytest.mark.parametrize("fill", [texture_copy_fill, patch_fill])
def test_copy_strategies_use_only_clean_allowed_donors(fill):
    image = np.full((32, 32, 3), [200, 20, 20], dtype=np.uint8)
    image[:8] = [25, 130, 40]
    hole = np.zeros((32, 32), dtype=bool)
    hole[15:21, 15:21] = True
    image[hole] = [230, 230, 230]
    donors = np.zeros_like(hole)
    donors[:8] = True
    filled = fill(image, hole, donor_mask=donors, seed=3)
    assert np.all(filled[hole] == [25, 130, 40])
    assert np.array_equal(filled[~hole], image[~hole])
    assert np.all(image[hole] == [230, 230, 230])


def test_patch_fill_shrinks_to_sparse_donors_without_copying_roof():
    image = np.full((9, 9, 3), 220, dtype=np.uint8)
    mask = np.ones((9, 9), dtype=bool)
    mask[0, 0] = False
    image[0, 0] = [30, 100, 50]
    filled = patch_fill(image, mask, patch_size=7)
    assert np.all(filled == [30, 100, 50])
    with pytest.raises(ValueError, match="no clean donor rectangle"):
        texture_copy_fill(image, mask)


@pytest.mark.parametrize("strategy", ["telea", "texture_copy", "patch_fill"])
def test_empty_and_border_footprints(strategy, building_scene):
    image, mask = building_scene
    empty = np.zeros_like(mask)
    pair = generate_synthetic_building_pair(image, empty, strategy=strategy)
    assert np.array_equal(pair["pre"], image)
    assert not pair["new_building"].any()
    border = np.zeros_like(mask)
    border[:6, :7] = 1
    image = image.copy()
    image[border != 0] = [230, 230, 230]
    pair = generate_synthetic_building_pair(image, border, strategy=strategy, feather_radius=1)
    assert np.array_equal(pair["new_building"], border)
    assert pair["pre"][border != 0].mean() < 140


@pytest.mark.parametrize("strategy", ["telea", "texture_copy", "patch_fill"])
def test_no_donors_rejected(strategy):
    image = np.full((8, 8, 3), 180, dtype=np.uint8)
    with pytest.raises(ValueError, match="donor"):
        SyntheticBuildingGenerator(strategy=strategy)(image, np.ones((8, 8), dtype=bool))


def test_seed_override_and_global_rng_independence(building_scene):
    image, mask = building_scene
    np.random.seed(29)
    expected = np.random.random(5)
    np.random.seed(29)
    generator = SyntheticBuildingGenerator(strategy="patch_fill", seed=4, max_candidates=5)
    pair = generator(image, mask, seed=8)
    NegativePairGenerator(seed=4)(image, "brightness")
    assert np.array_equal(np.random.random(5), expected)
    assert pair["metadata"]["seed"] == 8
    assert np.array_equal(pair["pre"], generator(image, mask, seed=8)["pre"])
    assert not np.array_equal(pair["pre"], generator(image, mask, seed=9)["pre"])


@pytest.mark.parametrize("bad_image", [np.zeros((3, 3, 3), dtype=float),
                                        np.zeros((3, 3), dtype=np.uint8),
                                        np.zeros((3, 3, 4), dtype=np.uint8)])
def test_image_contract_is_strict(bad_image):
    with pytest.raises(ValueError):
        SyntheticBuildingGenerator()(bad_image, np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        NegativePairGenerator()(bad_image)


def test_bad_masks_and_config_rejected(building_scene):
    image, mask = building_scene
    for invalid in (mask.astype(float), mask[..., None], mask[::2]):
        with pytest.raises(ValueError):
            SyntheticBuildingGenerator()(image, invalid)
    for config in ({"strategy": "black"}, {"seed": -1}, {"feather_radius": -1},
                   {"patch_size": 0}, {"inpaint_radius": float("nan")}, {"max_candidates": 0}):
        with pytest.raises(ValueError):
            SyntheticBuildingGenerator(**config)
    with pytest.raises(ValueError, match="copy strategies"):
        SyntheticBuildingGenerator()(image, mask, donor_mask=np.ones(mask.shape, dtype=bool))
    with pytest.raises(ValueError, match="donor"):
        SyntheticBuildingGenerator(strategy="patch_fill")(image, mask, donor_mask=np.zeros_like(mask))


@pytest.fixture
def forest_scene():
    pre = np.full((64, 64, 3), [40, 150, 50], dtype=np.uint8)
    post = pre.copy()
    post[20:40, 20:40] = [120, 80, 50]
    candidate = np.zeros((64, 64), dtype=np.uint8)
    candidate[10:50, 10:50] = 255
    return pre, post, candidate


def test_hansen_candidate_never_becomes_truth_even_with_strong_spectral_loss(forest_scene):
    pre, post, candidate = forest_scene
    pair = ForestLossRefiner()(pre, post, candidate)
    assert_pair_arrays(pair, candidate.shape)
    assert pair["proposed_mask"].sum() == 18 * 18
    assert not pair["tree_removal"].any()
    assert not pair["train_eligible"]
    assert not pair["metadata"]["train_eligible"]
    assert pair["metadata"]["label_source"] == "unlabeled"
    assert pair["metadata"]["review_status"] == "pending"
    assert pair["metadata"]["candidate_only"]
    assert not pair["metadata"]["hansen_direct_gt"]
    assert np.all(pair["proposed_mask"][candidate == 0] == 0)
    assert pair["confidence"].dtype == np.float32
    assert pair["confidence"][25, 25] > 0.8


def test_manual_approval_can_correct_heuristics_but_is_confined_to_reviewed_candidate(forest_scene):
    pre, post, candidate = forest_scene
    manual = np.zeros_like(candidate)
    # Reviewer can mark an event the RGB heuristic missed.
    manual[12:18, 12:18] = 1
    pair = ForestLossRefiner()(pre, post, candidate, manual_mask=manual,
                               review_decision="approve", reviewer="reviewer-17")
    assert np.array_equal(pair["tree_removal"], manual)
    assert pair["train_eligible"]
    assert pair["metadata"]["label_source"] == "manual_review"
    assert pair["metadata"]["reviewer"] == "reviewer-17"
    assert pair["metadata"]["training_scope"] == "review_mask_only"
    assert np.array_equal(pair["review_mask"], candidate != 0)
    assert not np.shares_memory(manual, pair["tree_removal"])
    outside = manual.copy()
    outside[0, 0] = 1
    with pytest.raises(ValueError, match="confined"):
        ForestLossRefiner()(pre, post, candidate, manual_mask=outside,
                            review_decision="approve", reviewer="reviewer-17")


def test_rejection_produces_reviewed_negative_only_with_reviewer(forest_scene):
    pre, post, candidate = forest_scene
    pair = refine_forest_loss(pre, post, candidate, review_decision="reject", reviewer="reviewer-2")
    assert pair["train_eligible"] and not pair["tree_removal"].any()
    assert pair["metadata"]["review_status"] == "rejected"
    for options in ({"review_decision": "approve", "reviewer": "a"},
                    {"review_decision": "reject"},
                    {"review_decision": "approve", "manual_mask": candidate},
                    {"manual_mask": candidate}, {"reviewer": "a"},
                    {"review_decision": "reject", "reviewer": "a", "manual_mask": candidate},
                    {"review_decision": "auto", "reviewer": "a"}):
        with pytest.raises(ValueError):
            refine_forest_loss(pre, post, candidate, **options)


def test_forest_unchanged_shadow_small_components_invalid_pixels_and_candidate_bounds(forest_scene):
    pre, _, candidate = forest_scene
    refiner = ForestLossRefiner()
    assert not refiner(pre, pre, candidate)["proposed_mask"].any()
    assert not refiner(pre, pre // 2, candidate)["proposed_mask"].any()
    shadow = pre.copy()
    shadow[20:40, 20:40] = 0
    assert not refiner(pre, shadow, candidate)["proposed_mask"].any()
    post = pre.copy()
    post[22:25, 22:25] = [120, 80, 50]
    post[:8, :8] = [120, 80, 50]  # Strong change outside candidate must not leak.
    assert not refiner(pre, post, candidate)["proposed_mask"].any()
    post[20:40, 20:40] = [120, 80, 50]
    valid = np.ones(candidate.shape, dtype=bool)
    valid[20:40, 20:40] = False
    result = refiner(pre, post, candidate, valid_mask=valid)
    assert not result["proposed_mask"].any()
    assert not result["confidence"][~valid].any()
    with pytest.raises(ValueError, match="confined"):
        refiner(pre, post, candidate, valid_mask=valid, manual_mask=(~valid),
                review_decision="approve", reviewer="human")


def test_forest_accepts_high_resolution_evidence_but_still_requires_manual_review(forest_scene):
    pre, post, candidate = forest_scene
    # Gray imagery provides no RGB vegetation evidence, but an external canopy model can.
    pre = np.full_like(pre, 90)
    post = np.full_like(post, 110)
    before = np.full(candidate.shape, 0.9, dtype=np.float32)
    after = before.copy()
    after[20:40, 20:40] = 0.05
    result = ForestLossRefiner()(pre, post, candidate, pre_vegetation=before, post_vegetation=after)
    assert result["proposed_mask"].sum() == 324
    assert not result["train_eligible"] and not result["tree_removal"].any()
    assert np.array_equal(before, np.full(candidate.shape, 0.9, dtype=np.float32))
    for invalid in (before[::2], before.astype(np.uint8), before * np.nan, before * 2):
        with pytest.raises(ValueError):
            ForestLossRefiner()(pre, post, candidate, pre_vegetation=invalid)
    with pytest.raises(ValueError, match="implicit resampling"):
        ForestLossRefiner()(pre, post, np.ones((2, 2), dtype=np.uint8))


def test_forest_empty_review_has_no_training_pixels(forest_scene):
    pre, post, candidate = forest_scene
    pair = refine_forest_loss(pre, post, np.zeros_like(candidate), review_decision="reject", reviewer="human")
    assert not pair["train_eligible"]
    assert not pair["review_mask"].any()


@pytest.mark.parametrize("kind", NEGATIVE_TYPES)
def test_all_nine_requested_negatives_are_deterministic_background_pairs(building_scene, kind):
    image, _ = building_scene
    original = image.copy()
    generator = NegativePairGenerator(seed=11)
    pair = generator(image, kind)
    assert_pair_arrays(pair, image.shape[:2])
    assert np.array_equal(pair["pre"], original)
    assert np.array_equal(image, original)
    assert not pair["new_building"].any() and not pair["tree_removal"].any()
    assert pair["metadata"]["kind"] == kind
    assert pair["metadata"]["train_eligible"]
    assert np.array_equal(pair["post"], generator(image, kind)["post"])
    assert pair["metadata"] == generator(image, kind)["metadata"]
    if kind == "exact_no_change":
        assert np.array_equal(pair["pre"], pair["post"])
    else:
        assert not np.array_equal(pair["pre"], pair["post"])


def test_section13_inventory_is_complete():
    assert set(NEGATIVE_TYPES) == {
        "exact_no_change", "brightness", "seasonal_color", "shadows", "vegetation_appearance",
        "slight_shift", "compression", "blur", "temporary_objects",
    }
    assert set(EXTRA_NEGATIVE_TYPES) == {"same_region_season", "roof_color", "agricultural", "road"}


@pytest.mark.parametrize("kind", ["roof_color", "agricultural", "road"])
def test_local_hard_negatives_require_roi_and_preserve_other_pixels(building_scene, kind):
    image, mask = building_scene
    with pytest.raises(ValueError, match="explicit roi_mask"):
        NegativePairGenerator()(image, kind)
    pair = generate_negative_pair(image, kind, roi_mask=mask, seed=6)
    assert np.array_equal(pair["post"][mask == 0], image[mask == 0])
    assert not np.array_equal(pair["post"][mask != 0], image[mask != 0])
    assert not pair["new_building"].any() and not pair["tree_removal"].any()


def test_shift_excludes_padding_and_preserves_exact_translated_content(building_scene):
    image, _ = building_scene
    pair = generate_negative_pair(image, "slight_shift", seed=11, max_shift=2)
    params = pair["metadata"]["parameters"]
    dx, dy = params["dx"], params["dy"]
    assert dx != 0 or dy != 0
    valid = pair["valid_mask"].astype(bool)
    yy, xx = np.nonzero(valid)
    assert np.array_equal(pair["post"][yy, xx], image[yy - dy, xx - dx])
    assert valid.sum() == (image.shape[0] - abs(dy)) * (image.shape[1] - abs(dx))
    assert params["training_scope"] == "valid_mask_only"


def test_jpeg_maintains_rgb_channel_order():
    image = np.full((24, 24, 3), [210, 30, 65], dtype=np.uint8)
    pair = generate_negative_pair(image, "compression", jpeg_quality=95)
    assert np.max(np.abs(pair["post"].astype(int) - image.astype(int))) <= 3
    assert pair["post"][..., 0].mean() > pair["post"][..., 2].mean()


def test_real_temporal_negatives_cannot_be_inferred_from_region_season_alone(building_scene):
    image, _ = building_scene
    metadata = {"region_pre": "tile-17", "region_post": "tile-17",
                "season_pre": "summer", "season_post": "summer"}
    changed = np.clip(image.astype(int) + 5, 0, 255).astype(np.uint8)
    with pytest.raises(ValueError, match="confirmed_no_change"):
        generate_negative_pair(image, "same_region_season", post=changed, **metadata)
    pair = generate_negative_pair(image, "same_region_season", post=changed,
                                  confirmed_no_change=True, **metadata)
    assert np.array_equal(pair["post"], changed)
    assert not pair["new_building"].any()
    assert pair["metadata"]["label_source"] == "reviewed_no_change"
    assert not pair["metadata"]["synthetic"]
    with pytest.raises(ValueError, match="matching"):
        generate_negative_pair(image, "same_region_season", post=changed, confirmed_no_change=True,
                               **{**metadata, "region_post": "different"})
    with pytest.raises(ValueError, match="matching"):
        generate_negative_pair(image, "same_region_season", post=changed, confirmed_no_change=True,
                               **{**metadata, "season_post": "winter"})
    with pytest.raises(ValueError, match="requires a reviewed post"):
        generate_negative_pair(image, "same_region_season", confirmed_no_change=True, **metadata)
    with pytest.raises(ValueError, match="post is allowed only"):
        generate_negative_pair(image, "brightness", post=changed)


@pytest.mark.parametrize("alias,canonical", [("same_image", "exact_no_change"), ("shadow", "shadows"),
                                             ("small_shift", "slight_shift"), ("seasonal", "seasonal_color")])
def test_negative_aliases(alias, canonical, building_scene):
    image, _ = building_scene
    pair = generate_negative_pair(image, alias, seed=7)
    reference = generate_negative_pair(image, canonical, seed=7)
    assert np.array_equal(pair["post"], reference["post"])
    assert pair["metadata"]["kind"] == canonical


def test_negative_bad_inputs_and_memory_independence(building_scene):
    image, mask = building_scene
    pair = generate_negative_pair(image)
    pair["post"][:] = 0
    assert image.any() and pair["pre"].any()
    with pytest.raises(ValueError, match="unknown"):
        generate_negative_pair(image, "new_construction")
    with pytest.raises(ValueError):
        generate_negative_pair(image, "brightness", roi_mask=mask.astype(float))
    for options in ({"seed": -1}, {"max_shift": 0}, {"jpeg_quality": 101}, {"jpeg_quality": 1.5}):
        with pytest.raises(ValueError):
            NegativePairGenerator(**options)


@pytest.mark.parametrize("kind", NEGATIVE_TYPES)
def test_tiny_images_are_supported_for_negatives(kind):
    pair = generate_negative_pair(np.full((1, 1, 3), [40, 150, 50], dtype=np.uint8), kind)
    assert_pair_arrays(pair, (1, 1))


def test_direct_telea_api_and_noncontiguous_input(building_scene):
    image, mask = building_scene
    sliced_image = image[:, ::-1]
    sliced_mask = mask[:, ::-1]
    pair = generate_synthetic_building_pair(sliced_image, sliced_mask, strategy="telea")
    assert np.array_equal(pair["post"], sliced_image)
    filled = telea_fill(sliced_image, sliced_mask, radius=3)
    assert filled.shape == image.shape and filled.dtype == np.uint8
