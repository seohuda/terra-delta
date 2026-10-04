"""Offline candidate generation never promotes spectral/static evidence to GT."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

from terradelta.data.dataset import read_manifest
from terradelta.data.pairing import training_eligibility


def test_real_source_plan_generates_review_evidence_without_labels_or_approval(tmp_path):
    spec = importlib.util.spec_from_file_location("pilot_candidates", Path(__file__).parents[1] / "scripts/pilot_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "data"
    (root / "metadata").mkdir(parents=True)
    transform = from_origin(300000, 4300002, .6, .6)
    bounds = transform_bounds("EPSG:26918", "EPSG:4326", 300000, 4300002-156, 300000+156, 4300002)
    grid = np.indices((260, 260)).sum(axis=0).astype(np.uint8)
    arrays = np.stack([grid, np.full_like(grid, 170), grid//2, np.full_like(grid, 220)])
    assets = []
    for role, year in (("pre", 2018), ("post", 2021)):
        path = root / f"{role}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=260, height=260, count=4, dtype="uint8",
                           crs="EPSG:26918", transform=transform, photometric="MINISBLACK") as ds:
            ds.write(arrays)
        assets.append({"id": role, "year": year, "source": "NAIP", "url": f"https://example.invalid/{role}.tif", "path": str(path)})
    loss_path = root / "loss.tif"
    with rasterio.open(loss_path, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(bounds[0]-.005, bounds[3]+.005, .0003, .0003)) as ds:
        ds.write(np.full((1, 64, 64), 20, np.uint8))
    footprint_path = root / "footprints.gz"
    with gzip.open(footprint_path, "wt") as handle:
        handle.write(json.dumps({"type": "Feature", "geometry": mapping(box(*bounds)), "properties": {}}) + "\n")
    sources = assets + [{"id": "lossyear_40N_080W", "source": "Hansen_GFC", "region": "40N_080W", "path": str(loss_path)},
                        {"id": "microsoft_032010032", "path": str(footprint_path)}]
    files = {
        "acquisition-plan.json": {"naip": [{"region": "md_fixture", "bounds": bounds, "assets": assets,
            "raster_headers": [{"id": a["id"], "acquired": f"{a['year']}-07-01T00:00:00Z"} for a in assets]}]},
        "download-report.json": {"completed_at_utc": "fixture", "assets": sources},
        "microsoft-selected.json": {"wanted_quadkeys": {"md_fixture": "032010032"}},
    }
    for name, payload in files.items():
        (root / "metadata" / name).write_text(json.dumps(payload))
    output = tmp_path / "candidates"
    module.build(root, output)
    rows = read_manifest(output / "candidates.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["label_status"] == "candidate" and row["confidence"] == "UNREVIEWED"
    assert not training_eligibility(row)[0] and row["synthetic"] == "False"
    assert not row["new_building"] and not row["tree_removal"]
    folder = Path(row["pre"]).parent
    assert (folder / "hansen_candidate.png").exists()
    assert not (folder / "mask.png").exists() and not (folder / "tree_removal.png").exists()
    with pytest.raises(FileExistsError):
        module.build(root, output)
