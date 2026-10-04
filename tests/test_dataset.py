"""Offline loader, label, augmentation, geographic split and local raster contracts."""

import csv
import json

import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds, from_origin

from terradelta.data.dataset import ChangeDataset, read_manifest
from terradelta.data.pairing import (
    discover_temporal_pairs,
    filter_training_rows,
    iter_pair_tiles,
    pair_grid,
    training_eligibility,
    write_manifest,
)
from terradelta.data.split import assert_no_spatial_leak, spatial_groups, split_manifest
from terradelta.data.transforms import PairedTransform, build_transforms
from terradelta.utils.geo import project_bounds


@pytest.fixture
def sample(tmp_path):
    folder = tmp_path / "samples" / "tile-a"
    folder.mkdir(parents=True)
    pre = np.full((8, 10, 3), [51, 102, 153], dtype=np.uint8)
    post = np.full_like(pre, [204, 153, 102])
    building = np.zeros((8, 10), dtype=np.uint8)
    tree = np.zeros_like(building)
    building[1:3, 2:5] = 255
    tree[5:7, 6:9] = 1
    for name, array in (("pre", pre), ("post", post), ("new_building", building), ("tree_removal", tree)):
        Image.fromarray(array).save(folder / f"{name}.png")
    return folder, pre, post, building, tree


def test_partial_review_scope_ignores_unreviewed_pixels_and_returns_auxiliary_masks(sample, tmp_path):
    folder, _, _, building, tree = sample
    valid = np.ones(building.shape, np.uint8)
    valid[:, 0] = 0
    review = np.zeros_like(valid)
    review[1:7, :8] = 255
    for key, array in (("valid_mask", valid), ("review_mask", review)):
        Image.fromarray(array).save(folder / f"{key}.png")
    row = {"id": "partial", **{key: str(folder / f"{key}.png") for key in
           ("pre", "post", "new_building", "tree_removal", "valid_mask", "review_mask")}}
    path = tmp_path / "partial.csv"
    write_manifest([row], path)
    assert read_manifest(path)[0]["review_mask"] == row["review_mask"]
    result = ChangeDataset(path, return_auxiliary=True)[0]
    scope = (valid != 0) & (review != 0)
    expected = (building > 0).astype(np.int64) + 2 * (tree > 0)
    expected[~scope] = -100
    assert result["has_labels"]  # Class annotation presence does not imply full review scope.
    assert np.array_equal(result["mask"].numpy(), expected)
    assert result["valid_mask"].dtype == torch.bool
    assert np.array_equal(result["valid_mask"].numpy(), scope)
    assert result["auxiliary_mask"].shape == (2, 8, 10)
    assert result["auxiliary_mask"].dtype == torch.bool
    assert np.array_equal(result["auxiliary_mask"][0].numpy(), expected == 1)
    assert np.array_equal(result["auxiliary_mask"][1].numpy(), expected == 2)
    assert not result["auxiliary_mask"][:, ~result["valid_mask"]].any()
    assert "auxiliary_mask" not in ChangeDataset(path)[0]


def test_geometric_augmentation_preserves_int16_ignore_sentinel():
    mask = np.zeros((24, 24), np.int16)
    mask[:6] = -100
    mask[8:12, 5:10] = 1
    mask[14:20, 13:18] = 2
    image = np.random.default_rng(1).integers(20, 200, (24, 24, 3), dtype=np.uint8)
    transform = PairedTransform(affine_probability=1, photometric_probability=0)
    for seed in range(6):
        result = transform(pre=image, post=image, mask=mask, seed=seed)
        assert result["mask"].dtype == np.int16
        assert set(np.unique(result["mask"])) == {-100, 0, 1, 2}
        assert np.array_equal(result["pre"], result["post"])
    assert np.all(mask[:6] == -100)


@pytest.mark.parametrize("bad_scope", ["missing", "shape", "values"])
def test_partial_scope_masks_are_validated(sample, tmp_path, bad_scope):
    folder, _, _, building, _ = sample
    scope_path = folder / "review_mask.png"
    if bad_scope != "missing":
        array = building[::2] if bad_scope == "shape" else np.full_like(building, 17)
        Image.fromarray(array).save(scope_path)
    row = {"id": "partial", **{key: str(folder / f"{key}.png")
           for key in ("pre", "post", "new_building", "tree_removal")}, "review_mask": str(scope_path)}
    path = tmp_path / "partial.csv"
    write_manifest([row], path)
    with pytest.raises((ValueError, FileNotFoundError)):
        ChangeDataset(path)[0]


@pytest.mark.parametrize("name", ["ce", "weighted_ce", "dice", "ce_dice", "focal", "focal_dice"])
def test_partial_label_losses_ignore_unreviewed_logits_without_optimizer_or_backward(name):
    from terradelta.training.losses import SegmentationLoss

    criterion = SegmentationLoss(name, class_weights=[1, 2, 3])
    logits = torch.zeros((1, 3, 4, 4))
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[:, :2] = -100
    target[:, 2, 1] = 1
    changed = logits.clone()
    changed[:, 1, :2] = 1000
    changed[:, 2, :2] = -1000
    torch.testing.assert_close(criterion(logits, target), criterion(changed, target))
    empty = criterion(changed, torch.full_like(target, -100))
    assert torch.isfinite(empty) and empty.item() == 0


def test_validation_rejects_partial_scope_with_mock_predictions_only(monkeypatch):
    from terradelta.inference import predictor
    from terradelta.training.validation import validate_model

    class MockPredictor:
        def __init__(self, *args, **kwargs):
            pass

        def predict_batch(self, images):
            probabilities = np.zeros((len(images), 3, *images.shape[-2:]), np.float32)
            probabilities[:, 0] = 1
            return probabilities

    monkeypatch.setattr(predictor, "Predictor", MockPredictor)
    valid = torch.ones((4, 4), dtype=torch.bool)
    valid[0] = False
    sample = {"id": "partial", "image": torch.zeros((6, 4, 4)), "mask": torch.zeros((4, 4), dtype=torch.long),
              "has_labels": True, "valid_mask": valid}
    model = torch.nn.Identity()
    with pytest.raises(ValueError, match="fully annotated"):
        validate_model(model, [sample], {})
    assert model.training  # Failure also restores the caller's mode.


def csv_file(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_folder_loader_normalizes_each_timestamp_and_preserves_labels(sample):
    folder, pre, post, building, tree = sample
    dataset = ChangeDataset(folder)
    item = dataset[0]
    assert len(dataset) == 1 and item["id"] == "tile-a" and item["has_labels"]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for name, image in (("pre", pre), ("post", post)):
        expected = (torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255 - mean) / std
        torch.testing.assert_close(item[name], expected)
        assert item[name].dtype == torch.float32 and item[name].is_contiguous()
    assert item["image"].shape == (6, 8, 10)
    torch.testing.assert_close(item["image"][:3], item["pre"])
    torch.testing.assert_close(item["image"][3:], item["post"])
    assert item["mask"].dtype == torch.int64
    assert np.array_equal(item["mask"].numpy(), (building > 0) + 2 * (tree > 0))
    assert ChangeDataset(folder.parent)[0]["id"] == "tile-a"


def test_manifest_roundtrip_resolves_relative_paths_and_retains_provenance(sample, tmp_path, monkeypatch):
    folder, *_ = sample
    row = {"id": "tile-a", **{key: str(folder / f"{key}.png")
                              for key in ("pre", "post", "new_building", "tree_removal")},
           "source": "NAIP+FEMA", "region_id": "md-1", "license_status": "public_domain",
           "year_pre": 2019, "year_post": 2021, "bounds": json.dumps([0, 0, 10, 8]),
           "crs": "EPSG:32618", "reviewer": "offline-fixture", "label_status": "reviewed"}
    path = tmp_path / "manifests" / "approved.csv"
    write_manifest([row], path)
    with path.open() as handle:
        stored = next(csv.DictReader(handle))
    assert stored["pre"].startswith("../samples/")
    monkeypatch.chdir(tmp_path.parent)
    loaded = read_manifest(path)[0]
    assert loaded["pre"] == row["pre"] and loaded["reviewer"] == "offline-fixture"
    assert loaded["bounds"] == row["bounds"] and loaded["year_post"] == "2021"
    assert ChangeDataset(path)[0]["has_labels"]
    empty = tmp_path / "empty.csv"
    write_manifest([], empty)
    assert read_manifest(empty) == []
    with pytest.raises(ValueError, match="no samples"):
        ChangeDataset(empty)


@pytest.mark.parametrize("rows,fields,match", [
    ([{"id": "a", "pre": "a"}], ["id", "pre"], "missing columns"),
    ([{"id": " ", "pre": "a", "post": "b"}], None, "nonempty and unique"),
    ([{"id": "a", "pre": "a", "post": "b"}] * 2, None, "nonempty and unique"),
])
def test_bad_manifest_rejected(tmp_path, rows, fields, match):
    path = csv_file(tmp_path / "bad.csv", rows, fields)
    with pytest.raises(ValueError, match=match):
        read_manifest(path)


@pytest.mark.parametrize("absence", [{"tree_removal": "absent"}, {"tree_removal_absent": "true"}])
def test_verified_class_absence_is_distinct_from_unlabeled(sample, tmp_path, absence):
    folder, *_ = sample
    row = {"id": "a", "pre": str(folder / "pre.png"), "post": str(folder / "post.png"),
           "new_building": str(folder / "new_building.png"), "tree_removal": "", **absence}
    path = csv_file(tmp_path / "pairs.csv", [row])
    item = ChangeDataset(path)[0]
    assert item["has_labels"] and 2 not in item["mask"]
    row = {**row, "tree_removal": "", "tree_removal_absent": ""}
    path = csv_file(path, [row])
    with pytest.raises(ValueError, match="Missing tree_removal"):
        ChangeDataset(path)[0]
    item = ChangeDataset(path, require_masks=False)[0]
    assert not item["has_labels"] and 2 not in item["mask"]


@pytest.mark.parametrize("corruption,match", [
    ("overlap", "Overlapping"), ("values", "0/1/255"), ("shape", "Mask shape"),
    ("rgb", "single-channel"), ("pair_shape", "Pair dimensions"), ("missing", "Missing"),
])
def test_bad_annotations_and_pairs_fail_closed(sample, corruption, match):
    folder, pre, _, building, _ = sample
    if corruption == "overlap":
        Image.fromarray(building).save(folder / "tree_removal.png")
    elif corruption == "values":
        Image.fromarray(np.full(building.shape, 2, np.uint8)).save(folder / "new_building.png")
    elif corruption == "shape":
        Image.fromarray(building[::2]).save(folder / "new_building.png")
    elif corruption == "rgb":
        Image.fromarray(pre).save(folder / "new_building.png")
    elif corruption == "pair_shape":
        Image.fromarray(pre[::2]).save(folder / "post.png")
    else:
        (folder / "tree_removal.png").unlink()
    with pytest.raises(ValueError, match=match):
        ChangeDataset(folder)[0]


def test_unlabeled_folder_is_inference_only(sample):
    folder, *_ = sample
    for key in ("new_building", "tree_removal"):
        (folder / f"{key}.png").unlink()
    item = ChangeDataset(folder, require_masks=False)[0]
    assert not item["has_labels"] and not item["mask"].any()


@pytest.mark.parametrize("corruption", ["mask_values", "mask_shape", "image_shape"])
def test_transform_output_contract_is_enforced(sample, corruption):
    def broken(**arrays):
        if corruption == "mask_values":
            arrays["mask"] = np.full_like(arrays["mask"], 3)
        elif corruption == "mask_shape":
            arrays["mask"] = arrays["mask"][::2]
        else:
            arrays["post"] = arrays["post"][::2]
        return arrays
    with pytest.raises(ValueError, match="Transform"):
        ChangeDataset(sample[0], transform=broken)[0]


@pytest.mark.parametrize("shape", [(16, 16), (12, 20)])
def test_geometry_replays_both_timestamps_and_discrete_labels(shape):
    mask = np.zeros(shape, np.uint8)
    mask[2:5, 3:7] = 1
    mask[7:10, 8:11] = 2
    image = np.repeat((mask * 90)[..., None], 3, axis=2)
    original = image.copy()
    transform = PairedTransform(seed=12, photometric_probability=0, affine_probability=0)
    for seed in range(8):
        result = transform(pre=image, post=image, mask=mask, seed=seed)
        assert result["pre"].shape == (*shape, 3)
        assert np.array_equal(result["pre"], result["post"])
        assert np.array_equal(result["pre"][..., 0], result["mask"] * 90)
        assert result["mask"].flags.c_contiguous
    assert np.array_equal(image, original)


def test_affine_and_independent_appearance_are_reproducible_without_global_rng():
    image = np.random.default_rng(19).integers(20, 220, (24, 24, 3), dtype=np.uint8)
    mask = np.zeros((24, 24), np.uint8)
    mask[4:9, 7:12] = 2
    np.random.seed(31)
    expected = np.random.random(5)
    np.random.seed(31)
    transform = PairedTransform(affine_probability=1, photometric_probability=0)
    warped = transform(pre=image, post=image, mask=mask, seed=5)
    assert np.array_equal(warped["pre"], warped["post"])
    assert set(np.unique(warped["mask"])) <= {0, 2}
    photo = PairedTransform(geometric=False, photometric_probability=1)
    result = photo(pre=image, post=image, mask=mask, seed=8)
    repeat = photo(pre=image, post=image, mask=mask, seed=8)
    assert not np.array_equal(result["pre"], result["post"])
    for key in result:
        assert np.array_equal(result[key], repeat[key])
    assert np.array_equal(result["mask"], mask)
    assert np.array_equal(np.random.random(5), expected)
    photo.reseed(9)
    first = photo(pre=image, post=image, mask=mask)
    photo.reseed(9)
    assert np.array_equal(first["pre"], photo(pre=image, post=image, mask=mask)["pre"])
    inference = build_transforms({"augmentation": {"photometric_probability": 1}}, training=False)
    assert np.array_equal(inference(pre=image, post=image, mask=mask)["pre"], image)


def test_disabled_augmentation_config_preserves_inputs():
    """Regression: enabled=False must disable the configured training transforms."""
    image = np.random.default_rng(2).integers(0, 255, (12, 12, 3), dtype=np.uint8)
    mask = np.zeros((12, 12), np.uint8)
    transform = build_transforms({"augmentation": {"enabled": False, "photometric_probability": 1}})
    result = transform(pre=image, post=image, mask=mask)
    assert np.array_equal(result["pre"], image)
    assert np.array_equal(result["post"], image)


@pytest.mark.parametrize("row", [
    {"source": "LEVIR-CD", "license_status": "commercial_ok"},
    {"source": "NAIP+AIHub", "license_status": "public_domain"},
    {"source": "NAIP", "license_status": "unknown"},
    {"source": "", "license_status": "public_domain"},
    {"source": "Hansen", "license_status": "cc_by_4.0"},
    {"source": "NAIP", "license_status": "public_domain", "label_source": "hansen"},
    {"source": "NAIP", "license_status": "public_domain", "label_status": "weak_unreviewed"},
])
def test_unapproved_sources_and_candidate_labels_are_excluded(row):
    assert not training_eligibility(row)[0]
    approved, excluded = filter_training_rows([{"id": "bad", **row}])
    assert not approved and excluded[0]["exclusion_reason"]


def test_approved_manifest_filter_does_not_modify_input():
    row = {"id": "a", "source": "NAIP+FEMA", "license_status": "public_domain", "label_status": "reviewed"}
    approved, excluded = filter_training_rows([row])
    assert approved == [row] and not excluded and approved[0] is not row


def test_temporal_discovery_groups_regions_and_limits_year_gap():
    records = [{"region_id": region, "year": year, "path": f"{region}-{year}.tif"}
               for region in ("a", "b") for year in (2018, 2020, 2023)]
    pairs = discover_temporal_pairs(records, min_year_gap=2, max_year_gap=3)
    assert len(pairs) == 4
    assert all(p["pre"]["region_id"] == p["post"]["region_id"] for p in pairs)
    assert all(2 <= p["post"]["year"] - p["pre"]["year"] <= 3 for p in pairs)
    with pytest.raises(ValueError, match="region_id"):
        discover_temporal_pairs([{"year": 2020, "path": "x"}])


def split_rows():
    return [{"id": f"r{i}-{j}", "region_id": f"r{i}", "state": "MD" if i < 2 else "VA",
             "year_pre": 2017, "year_post": 2023 if i == 3 and j == 1 else 2019}
            for i in range(4) for j in range(2)]


@pytest.mark.parametrize("strategy", ["random", "region", "state", "temporal"])
def test_splits_are_repeatable_complete_and_geographically_disjoint(strategy):
    rows = split_rows()
    options = {"strategy": strategy, "seed": 7, "temporal_year": 2022}
    parts = split_manifest(rows, **options)
    assert parts == split_manifest(rows, **options)
    assert parts["train"] and parts["val"]
    assert {r["id"] for r in parts["train"] + parts["val"]} == {r["id"] for r in rows}
    assert_no_spatial_leak(**{"train_rows": parts["train"], "val_rows": parts["val"]})
    if strategy == "state":
        assert {r["state"] for r in parts["train"]}.isdisjoint(r["state"] for r in parts["val"])
    if strategy == "temporal":
        assert {r["region_id"] for r in parts["val"]} == {"r3"}
        assert len(parts["val"]) == 2  # Old image of held-out region stays held out.


def test_shared_source_image_across_temporal_roles_and_parent_derivatives_stay_together():
    rows = [{"id": "a", "region_id": "a", "post": "shared.tif"},
            {"id": "b", "region_id": "b", "pre": "shared.tif", "parent_id": "family"},
            {"id": "c", "region_id": "c", "parent_id": "family"},
            {"id": "d", "region_id": "d"}]
    assert spatial_groups(rows) == [[0, 1, 2], [3]]
    with pytest.raises(ValueError, match="share"):
        assert_no_spatial_leak(rows[:1], rows[1:])


def test_touching_and_buffered_tiles_group_across_crs():
    rows = [{"id": "a", "region_id": "a", "bounds": [500000, 4300000, 500010, 4300010], "crs": "EPSG:32618"},
            {"id": "b", "region_id": "b", "bounds": [500010, 4300000, 500020, 4300010], "crs": "EPSG:32618"},
            {"id": "c", "region_id": "c", "bounds": [500022, 4300000, 500032, 4300010], "crs": "EPSG:32618"}]
    assert spatial_groups(rows) == [[0, 1], [2]]
    assert spatial_groups(rows, buffer_m=3) == [[0, 1, 2]]
    remote = {"id": "d", "region_id": "d", "bounds": project_bounds(rows[0]["bounds"], "EPSG:32618", "EPSG:4326"),
              "crs": "EPSG:4326"}
    assert spatial_groups([rows[0], remote]) == [[0, 1]]


@pytest.mark.parametrize("rows,options,match", [
    ([{"id": "a"}, {"id": "b"}], {}, "region_id"),
    ([{"region_id": "a"}, {"region_id": "a"}], {}, "two disconnected"),
    ([{"region_id": "a"}, {"region_id": "b"}], {"strategy": "state"}, "state"),
    ([{"region_id": "a"}, {"region_id": "b"}], {"strategy": "temporal"}, "temporal_year"),
    ([{"region_id": "a"}, {"region_id": "b"}], {"buffer_m": 1}, "bounds and CRS"),
    ([{"region_id": "a"}, {"region_id": "b"}], {"val_fraction": 1}, "between"),
    ([{"region_id": "a", "bounds": [0, 0, 1, 1], "crs": "EPSG:32618"}, {"region_id": "b"}], {}, "All rows"),
])
def test_unsafe_split_metadata_rejected(rows, options, match):
    with pytest.raises(ValueError, match=match):
        split_manifest(rows, **options)


@pytest.mark.parametrize("pre,post", [(2020, 2020), (2022, 2021), ("unknown", 2021), (None, 2021)])
def test_temporal_split_requires_ordered_numeric_acquisition_years(pre, post):
    rows = split_rows()
    rows[0] = {**rows[0], "year_pre": pre, "year_post": post}
    with pytest.raises(ValueError, match="numeric year_pre < year_post"):
        split_manifest(rows, strategy="temporal", temporal_year=2022)


def raster(path, *, crs="EPSG:32618", transform=None, shape=(10, 12), dtype="uint8", nodata=None, value=80):
    data = np.full((4, *shape), value, dtype=dtype)
    if nodata is not None:
        data[:, :4, :4] = nodata
    with rasterio.open(path, "w", driver="GTiff", width=shape[1], height=shape[0], count=4,
                       dtype=dtype, crs=crs, transform=transform or from_origin(500000, 4300010, 1, 1),
                       nodata=nodata, alpha="NO") as handle:
        handle.write(data)
        # NAIP band 4 is NIR; GDAL's default four-band GeoTIFF interpretation is RGBA.
        handle.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined)
    return path


def test_local_tiling_discards_edges_preserves_common_grid_and_nir(tmp_path):
    pre, post = raster(tmp_path / "pre.tif"), raster(tmp_path / "post.tif", value=120)
    tiles = list(iter_pair_tiles(pre, post, tile_size=4, bands=(1, 2, 3, 4), max_tiles=6))
    assert len(tiles) == 6  # 12x10 => six complete 4x4 tiles; bottom two rows discarded.
    assert {(t["row"], t["col"]) for t in tiles} == {(r, c) for r in (0, 4) for c in (0, 4, 8)}
    for tile in tiles:
        assert tile["pre"].shape == (4, 4, 4) and tile["valid"].all()
        assert np.all(tile["pre"] == 80) and np.all(tile["post"] == 120)
        assert tile["transform"] * (0, 0) == (500000 + tile["col"], 4300010 - tile["row"])


def test_tiling_reprojects_second_timestamp_and_excludes_nodata(tmp_path):
    pre = raster(tmp_path / "pre.tif", shape=(8, 8), nodata=0)
    bounds = project_bounds((500000, 4300002, 500008, 4300010), "EPSG:32618", "EPSG:3857")
    post = raster(tmp_path / "post.tif", crs="EPSG:3857", shape=(16, 16),
                  transform=from_bounds(*bounds, 16, 16), value=120)
    tiles = list(iter_pair_tiles(pre, post, tile_size=4, min_valid_fraction=0.9))
    assert len(tiles) == 3
    assert all(t["valid"].all() and np.all(t["post"] == 120) for t in tiles)
    with rasterio.open(pre) as a, rasterio.open(pre) as b:
        grid = pair_grid(a, b, bounds=(500004, 4300002, 500008, 4300010), bounds_crs="EPSG:32618")
    assert grid["width"] == 4 and grid["height"] == 8


@pytest.mark.parametrize("options,match", [
    ({"dst_crs": "EPSG:4326"}, "projected in meters"),
    ({"resolution": 0}, "positive"), ({"bands": (5,)}, "band is absent"),
    ({"max_tiles": 1}, "max_tiles"), ({"max_grid_pixels": 1}, "pixel limit"),
    ({"bounds": [500000, 4300000, 500010, 4300010]}, "bounds_crs"),
])
def test_tiling_rejects_unsafe_grids_and_bounds(tmp_path, options, match):
    pre, post = raster(tmp_path / "pre.tif"), raster(tmp_path / "post.tif")
    with pytest.raises(ValueError, match=match):
        list(iter_pair_tiles(pre, post, tile_size=4, **options))


def test_tiling_requires_local_uint8_georeferenced_overlapping_rasters(tmp_path):
    pre = raster(tmp_path / "pre.tif")
    with pytest.raises(ValueError, match="local files"):
        list(iter_pair_tiles("https://example.invalid/image.tif", pre))
    bad = raster(tmp_path / "bad.tif", dtype="uint16")
    with pytest.raises(ValueError, match="uint8"):
        list(iter_pair_tiles(pre, bad))
    remote = raster(tmp_path / "remote.tif", transform=from_origin(600000, 4300010, 1, 1))
    with pytest.raises(ValueError, match="increasing"):
        list(iter_pair_tiles(pre, remote))
    unknown = raster(tmp_path / "unknown.tif", crs=None)
    with pytest.raises(ValueError, match="CRS metadata"):
        list(iter_pair_tiles(pre, unknown))
