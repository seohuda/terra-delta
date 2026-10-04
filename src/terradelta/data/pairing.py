"""License-gated manifests and georeferenced local temporal image pairing."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from terradelta.utils.geo import project_bounds, validate_bounds
from .dataset import MANIFEST_FIELDS

# Explicit legal review status. Free access is not evidence of commercial rights.
APPROVED_LICENSE_STATUSES = {"public_domain", "cc_by_4.0", "commercial_ok", "verified_commercial"}
EXCLUDED_SOURCES = {"levir", "levir_cd", "levir_cd+", "aihub", "ai_hub"}


def training_eligibility(row):
    """Return (eligible, reason); unknown/unreviewed labels never enter training."""
    sources = {part.strip().lower().replace("-", "_") for part in str(row.get("source", "")).split("+")}
    if sources & EXCLUDED_SOURCES:
        return False, "source excluded pending competition-compatible rights"
    if not str(row.get("source", "")).strip():
        return False, "missing source provenance"
    if str(row.get("license_status", "unknown")).lower() not in APPROVED_LICENSE_STATUSES:
        return False, "commercial license not verified"
    # Hansen alone has no high-resolution change labels.
    if sources & {"hansen", "gfc", "hansen_gfc"} or row.get("label_source") == "hansen":
        return False, "Hansen is candidate mining only"
    if str(row.get("label_status", "")).lower() in {"candidate", "weak_unreviewed", "unreviewed"}:
        return False, "labels require high-resolution review"
    if row.get("training_scope") == "review_mask_only" and not row.get("review_mask"):
        return False, "partial weak labels require a review_mask path"
    return True, "approved"


def filter_training_rows(rows):
    approved, excluded = [], []
    for row in rows:
        eligible, reason = training_eligibility(row)
        if eligible:
            approved.append(dict(row))
        else:
            excluded.append({**row, "exclusion_reason": reason})
    return approved, excluded


def write_manifest(rows, path):
    """Write path fields relative to the manifest while retaining extra metadata."""
    path = Path(path).resolve()
    rows = list(rows)
    fields = list(MANIFEST_FIELDS) + sorted(set().union(*(row.keys() for row in rows)) - set(MANIFEST_FIELDS))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for original in rows:
            row = dict(original)
            for key in ("pre", "post", "new_building", "tree_removal", "valid_mask", "review_mask"):
                value = row.get(key)
                if value and str(value).lower() != "absent" and Path(value).is_absolute():
                    import os
                    row[key] = os.path.relpath(value, path.parent)
            writer.writerow(row)


def discover_temporal_pairs(records, *, min_year_gap=1, max_year_gap=None):
    """Pair records from the same region; never pair on year alone.

    Records carry ``region_id``, ``year``, and ``path``. Spatial overlap is checked
    by ``iter_pair_tiles`` against raster CRS/extent before data are emitted.
    """
    groups = {}
    for record in records:
        if not record.get("region_id"):
            raise ValueError("Temporal discovery requires region_id")
        groups.setdefault(str(record["region_id"]), []).append(record)
    pairs = []
    for region_id, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: (int(row["year"]), str(row["path"])))
        for i, pre in enumerate(ordered):
            for post in ordered[i + 1:]:
                gap = int(post["year"]) - int(pre["year"])
                if gap >= min_year_gap and (max_year_gap is None or gap <= max_year_gap):
                    pairs.append({"region_id": region_id, "pre": pre, "post": post})
    return pairs


def pair_grid(pre, post, *, resolution=1.0, dst_crs=None, bounds=None, bounds_crs=None):
    """Intersection grid in projected meters, inward snapped to complete pixels."""
    if not pre.crs or not post.crs:
        raise ValueError("Both rasters require CRS metadata")
    dst_crs = CRS.from_user_input(dst_crs or pre.crs)
    if not dst_crs.is_projected or any(abs(axis.unit_conversion_factor - 1.0) > 1e-8 for axis in dst_crs.axis_info[:2]):
        raise ValueError("Target CRS must be projected in meters for target resolution")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution must be positive meters/pixel")
    extents = [project_bounds(ds.bounds, ds.crs, dst_crs) for ds in (pre, post)]
    if bounds is not None:
        if bounds_crs is None:
            raise ValueError("AOI bounds require bounds_crs")
        extents.append(project_bounds(bounds, bounds_crs, dst_crs))
    west = math.ceil(max(b[0] for b in extents) / resolution) * resolution
    south = math.ceil(max(b[1] for b in extents) / resolution) * resolution
    east = math.floor(min(b[2] for b in extents) / resolution) * resolution
    north = math.floor(min(b[3] for b in extents) / resolution) * resolution
    validate_bounds((west, south, east, north))
    width, height = int(round((east - west) / resolution)), int(round((north - south) / resolution))
    return {"crs": dst_crs.to_string(), "transform": Affine(resolution, 0, west, 0, -resolution, north),
            "width": width, "height": height, "bounds": (west, south, east, north)}


def iter_pair_tiles(pre_path, post_path, *, resolution=1.0, dst_crs=None, bounds=None,
                    bounds_crs=None, tile_size=256, bands=(1, 2, 3), max_tiles=100,
                    max_grid_pixels=100_000_000, min_valid_fraction=1.0):
    """Stream bounded local raster windows with a common grid and validity mask.

    Complete tiles only: edge strips smaller than tile_size are discarded.
    No remote paths are accepted; fetch assets with guarded external downloaders.
    """
    for path in (pre_path, post_path):
        if not Path(path).is_file():
            raise ValueError(f"Pair tiling accepts downloaded local files only: {path}")
    if tile_size < 1 or tile_size > 4096 or max_tiles < 1 or not 0 <= min_valid_fraction <= 1:
        raise ValueError("Invalid tiling limits")
    with rasterio.open(pre_path) as pre, rasterio.open(post_path) as post:
        if any(b < 1 or b > min(pre.count, post.count) for b in bands):
            raise ValueError("Requested RGB/NIR band is absent")
        if any(ds.dtypes[b - 1] != "uint8" for ds in (pre, post) for b in bands):
            raise ValueError("NAIP imagery must be uint8; apply explicit calibrated scaling for other products")
        grid = pair_grid(pre, post, resolution=resolution, dst_crs=dst_crs, bounds=bounds, bounds_crs=bounds_crs)
        if grid["width"] * grid["height"] > max_grid_pixels:
            raise ValueError("Grid exceeds pixel limit; use a smaller AOI")
        n_tiles = (grid["width"] // tile_size) * (grid["height"] // tile_size)
        if n_tiles > max_tiles:
            raise ValueError(f"{n_tiles} tiles exceed explicit max_tiles={max_tiles}; narrow AOI")
        opts = {key: grid[key] for key in ("crs", "transform", "width", "height")}
        # Use a declared source alpha when present; otherwise request a VRT alpha
        # for coverage. An NIR data band is never silently used as validity.
        alpha_indexes = [next((i + 1 for i, interpretation in enumerate(ds.colorinterp)
                              if interpretation == ColorInterp.alpha), None) for ds in (pre, post)]
        if any(index in bands for index in alpha_indexes if index is not None):
            raise ValueError("Requested image band is tagged as alpha; inspect RGB/NIR metadata before tiling")
        with WarpedVRT(pre, **opts, resampling=Resampling.bilinear, add_alpha=alpha_indexes[0] is None) as pvrt, \
                WarpedVRT(post, **opts, resampling=Resampling.bilinear, add_alpha=alpha_indexes[1] is None) as qvrt:
            for row in range(0, grid["height"] - tile_size + 1, tile_size):
                for col in range(0, grid["width"] - tile_size + 1, tile_size):
                    window = Window(col, row, tile_size, tile_size)
                    a, b = pvrt.read(bands, window=window), qvrt.read(bands, window=window)
                    valid = (pvrt.read(alpha_indexes[0] or pvrt.count, window=window) > 0) & (qvrt.read(alpha_indexes[1] or qvrt.count, window=window) > 0)
                    if valid.mean() < min_valid_fraction:
                        continue
                    transform = pvrt.window_transform(window)
                    from rasterio.transform import array_bounds
                    yield {"pre": a.transpose(1, 2, 0), "post": b.transpose(1, 2, 0),
                           "valid": valid, "transform": transform, "crs": grid["crs"],
                           "bounds": array_bounds(tile_size, tile_size, transform), "row": row, "col": col}
