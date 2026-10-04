"""Audit failures must remain visible; fixtures never imply real-world readiness."""
import importlib.util
import json
from pathlib import Path
import socket

import numpy as np
from PIL import Image
import pytest
import torch

from terradelta.data.audit import audit_dataset, distribution, inspect_sample, inventory, leakage, provenance
from terradelta.data.dataset import ChangeDataset
from terradelta.data.pairing import training_eligibility, write_manifest
from terradelta.data.split import assert_no_spatial_leak, spatial_groups
from terradelta.external.common import cached_metadata_plan
from terradelta.inference.alignment import estimate_translation


@pytest.fixture
def rows(tmp_path):
    result = []
    for i in range(4):
        folder = tmp_path / f"tile_{i}"
        folder.mkdir()
        pre = np.random.default_rng(i).integers(30, 220, (256, 256, 3), dtype=np.uint8)
        post = np.roll(pre, 2, axis=1)
        mask = np.zeros((256, 256), np.uint8)
        if i in (1, 3):
            mask[10:20, 10:20] = 1
        if i in (2, 3):
            mask[30:40, 30:40] = 2
        for key, array in (("pre", pre), ("post", post), ("new_building", (mask == 1).astype(np.uint8) * 255),
                           ("tree_removal", (mask == 2).astype(np.uint8) * 255)):
            Image.fromarray(array).save(folder / f"{key}.png")
        result.append({"id": f"tile_{i}", **{k: str(folder / f"{k}.png") for k in
                       ("pre", "post", "new_building", "tree_removal")}, "region_id": f"r{i}",
                       "source": "synthetic_mock", "license_status": "commercial_ok", "label_status": "reviewed",
                       "year_pre": "2019", "year_post": "2021"})
    return result


def test_distribution_counts_both_classes_ignore_scope_and_components(rows):
    masks = {r["id"]: ChangeDataset([r])[0]["mask"].numpy() for r in rows}
    masks["tile_0"][:4] = -100
    result = distribution(masks)
    assert [result[k] for k in ("background_only", "new_building_only", "tree_removal_only", "both_classes")] == [1, 1, 1, 1]
    assert result["pixel_count"]["new_building"] == 200
    assert result["pixel_count"]["tree_removal"] == 200
    assert result["valid_pixel_count"] == 4 * 256**2 - 4 * 256
    assert sum(result["pixel_ratio"].values()) == pytest.approx(1)


def test_mock_or_missing_evidence_is_never_promoted_and_audit_gate_blocks_training(rows):
    audited = provenance(rows[0])
    assert audited["mock"] and not audited["audit_training_eligible"]
    assert audited["audit_status"] == "unverified"
    assert not training_eligibility(audited)[0]
    assert not provenance({**rows[0], "source": "NAIP"})["audit_training_eligible"]


@pytest.mark.parametrize("resolution", ["nan", "inf", "0", "-1"])
def test_nonfinite_resolution_cannot_be_verified_even_with_asserted_review(rows, resolution):
    row = {**rows[0], "source": "NAIP", "resolution_m": resolution,
           "pre_source": "NAIP", "post_source": "NAIP", "mask_source": "manual_review",
           "license_source": "provider_document", "source_url": "provider_asset", "reviewer": "fixture",
           "bounds": "[0,0,256,256]", "crs": "EPSG:32618", "provenance_status": "verified",
           "license_review_status": "verified"}
    assert not provenance(row)["audit_training_eligible"]


@pytest.mark.parametrize("field", ["pre_source_raster", "post_source_raster", "source_raster", "aoi_id", "location_id"])
def test_shared_raster_aoi_or_location_blocks_even_when_regions_and_years_differ(rows, field):
    a, b = dict(rows[0]), dict(rows[1])
    a[field], b[field] = "shared", "shared"
    b.update(year_pre="2021", year_post="2023")
    assert spatial_groups([a, b]) == [[0, 1]]
    with pytest.raises(ValueError, match="share"):
        assert_no_spatial_leak([a], [b])
    result = leakage([a, b], {a["id"]: "train", b["id"]: "val"})
    assert len(result["spatial_candidates"]) == len(result["temporal_candidates"]) == 1


def test_pre_post_raster_role_crossover_is_grouped(rows):
    a, b = dict(rows[0]), dict(rows[1])
    a["post_source_raster"], b["pre_source_raster"] = "raster_2021", "raster_2021"
    assert spatial_groups([a, b]) == [[0, 1]]


def test_missing_geography_does_not_become_no_leakage(rows):
    result = leakage(rows, {r["id"]: "val" if i == 0 else "train" for i, r in enumerate(rows)})
    assert result["spatial_candidates"] == []
    assert result["status"] == "unverified"


@pytest.mark.parametrize("failure", ["corrupt", "channels", "dimensions", "mask_values", "mask_shape", "empty_positive", "overlap", "nonfinite"])
def test_corruption_is_recorded_instead_of_silently_converting_or_crashing(rows, failure):
    row = dict(rows[0])
    if failure == "corrupt":
        Path(row["pre"]).write_bytes(b"not an image")
    elif failure == "channels":
        Image.fromarray(np.zeros((256, 256), np.uint8)).save(row["pre"])
    elif failure == "dimensions":
        Image.fromarray(np.zeros((128, 256, 3), np.uint8)).save(row["pre"])
    elif failure in {"mask_values", "mask_shape"}:
        shape = (256, 256) if failure == "mask_values" else (128, 256)
        Image.fromarray(np.full(shape, 17 if failure == "mask_values" else 0, np.uint8)).save(row["new_building"])
    elif failure == "empty_positive":
        row["new_building_present"] = "true"
    elif failure == "overlap":
        for key in ("new_building", "tree_removal"):
            Image.fromarray(np.ones((256, 256), np.uint8)).save(row[key])
    else:
        path = Path(row["pre"]).with_suffix(".tif")
        Image.fromarray(np.full((256, 256), np.nan, np.float32)).save(path)
        row["pre"] = str(path)
    _, _, issues, _, _ = inspect_sample(row)
    assert any(severity == "error" for severity, _ in issues)


def test_phase_estimate_recovers_translation_without_mutation():
    pre = np.random.default_rng(8).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    post = np.roll(np.roll(pre, 3, axis=1), -2, axis=0)
    original = pre.copy(), post.copy()
    result = estimate_translation(pre, post)
    assert result["reliable"]
    assert result["dx"] == pytest.approx(3, abs=.1)
    assert result["dy"] == pytest.approx(-2, abs=.1)
    assert np.array_equal(pre, original[0]) and np.array_equal(post, original[1])
    with pytest.raises(ValueError, match="nonconstant"):
        estimate_translation(np.zeros_like(pre), np.zeros_like(post))


def test_inventory_deduplicates_split_references_and_audit_excludes_generated_outputs(rows, tmp_path):
    write_manifest(rows, tmp_path / "all.csv")
    write_manifest(rows[:3], tmp_path / "train.csv")
    write_manifest(rows[3:], tmp_path / "val.csv")
    (tmp_path / "data_audit").mkdir()
    write_manifest(rows, tmp_path / "data_audit/audited.csv")
    result = inventory(tmp_path)
    assert result["unique_sample_folders"] == 4
    assert len(result["manifests"]) == 3
    assert result["image_mask_files"] == 16
    assert result["image_mask_bytes"] == sum(Path(r[k]).stat().st_size for r in rows for k in ("pre", "post", "new_building", "tree_removal"))


def test_cross_split_content_duplicate_and_bad_partition_prevent_readiness(rows, tmp_path, monkeypatch):
    rows[1]["pre"] = rows[0]["pre"]
    write_manifest(rows, tmp_path / "all.csv")
    write_manifest(rows[:1], tmp_path / "train.csv")
    write_manifest(rows[1:], tmp_path / "val.csv")
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("audit must stay offline"))
    result = audit_dataset(tmp_path / "all.csv", tmp_path / "audit", train_manifest=tmp_path / "train.csv", val_manifest=tmp_path / "val.csv")
    assert result["readiness"] == "NOT READY"
    leak = json.loads((tmp_path / "audit/split_audit.json").read_text())
    assert any(c["reason"] == "identical decoded image content" for c in leak["spatial_candidates"])
    assert result["training_eligible_samples"] == 0
    assert result["network"]["dataset_download_bytes"] == 0
    with pytest.raises(FileExistsError):
        audit_dataset(tmp_path / "all.csv", tmp_path / "audit")


def test_cached_plan_is_bounded_scoped_and_cannot_authorize_downloads(tmp_path):
    path = tmp_path / "plan.json"
    plan = {"source": "NAIP", "dry_run": True, "bounds": [0, 0, 1, 1], "years": [2021]}
    path.write_text(json.dumps(plan))
    assert cached_metadata_plan(path, source="NAIP", bounds=[0, 0, 1, 1], years=[2021])["new_download_bytes"] == 0
    for options in ({"source": "FEMA"}, {"bounds": [0, 0, 2, 2]}, {"years": [2019]}):
        with pytest.raises(ValueError):
            cached_metadata_plan(path, **{**dict(source="NAIP", bounds=[0, 0, 1, 1], years=[2021]), **options})
    path.write_text(json.dumps({**plan, "dry_run": False}))
    with pytest.raises(ValueError, match="metadata-only"):
        cached_metadata_plan(path, source="NAIP", bounds=[0, 0, 1, 1])
    path.write_bytes(b" " * 1_048_577)
    with pytest.raises(ValueError, match="1 MiB"):
        cached_metadata_plan(path, source="NAIP", bounds=[0, 0, 1, 1])


@pytest.mark.parametrize("name,source,args", [
    ("naip", "NAIP", ["--years", "2021"]),
    ("fema", "FEMA_USA_Structures", []),
])
def test_downloader_cached_dryrun_never_contacts_provider_or_creates_remote_path(name, source, args, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("download_" + name, Path(__file__).parents[1] / f"scripts/download_{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "discover_" + name, lambda **kwargs: pytest.fail("cached dry-run contacted provider"))
    path = tmp_path / "plan.json"
    plan = {"source": source, "dry_run": True, "bounds": [0, 0, 1, 1], "years": [2021]}
    if name == "fema":
        plan["endpoint"] = module.FEATURE_LAYER
    path.write_text(json.dumps(plan))
    remote = tmp_path / "nonexistent-remote-mount"
    report = module.main(["--bounds", "0", "0", "1", "1", "--metadata-plan", str(path), "--dry-run", "--output", str(remote), *args])
    assert report["output_path"] == str(remote) and not remote.exists()
    with pytest.raises(SystemExit):
        module.main(["--bounds", "0", "0", "1", "1", "--metadata-plan", str(path), "--download", *args])


def test_baseline_diagnostic_uses_inference_mode_and_never_constructs_optimizer(rows, tmp_path, monkeypatch):
    from terradelta.data.audit import baseline_analysis
    import terradelta.models.factory as factory
    import terradelta.models.checkpoint as checkpoints
    class EmptyModel(torch.nn.Module):
        def forward(self, image):
            assert not torch.is_grad_enabled()
            logits = torch.zeros((len(image), 3, *image.shape[-2:]))
            logits[:, 0] = 8
            return logits
    monkeypatch.setattr(factory, "build_model", lambda *a: EmptyModel())
    monkeypatch.setattr(checkpoints, "load_checkpoint", lambda *a: {"steps": 100})
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **k: pytest.fail("optimizer constructed"))
    masks = {r["id"]: ChangeDataset([r])[0]["mask"].numpy() for r in rows}
    validation, preview = baseline_analysis(rows, masks, {rows[0]["id"]}, "existing.pt", {"postprocess": {"mode": "argmax"}}, tmp_path)
    assert validation["metric_label"] == "approximate local metric"
    assert validation["total_validation_samples"] == 1 and validation["no_change_fp_rate"] == 0
    assert validation["optimizer_steps_executed"] == 0 and preview
    errors = json.loads((tmp_path / "error_analysis.json").read_text())
    assert errors["new_building_fn"] == ["tile_1", "tile_3"]
