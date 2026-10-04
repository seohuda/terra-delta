"""Deterministic connected geographic groups prevent region/neighbor leakage."""

from __future__ import annotations

import json
import math
import random

from pyproj import CRS
from shapely.geometry import box
from shapely.strtree import STRtree

from terradelta.utils.geo import project_bounds, validate_bounds


def _bounds(row):
    value = row.get("bounds")
    if value is None or value == "":
        return None
    return validate_bounds(json.loads(value) if isinstance(value, str) else value)


def spatial_groups(rows, *, strategy="region", buffer_m=0.0):
    """Join same regions, shared imagery/provenance and touching/buffered bounds.

    If bounds exist they must exist for ALL rows, with explicit CRS. Reprojection
    to a common projected metric CRS makes adjacency independent of source CRS.
    Without bounds, callers must give region ids encompassing adjacent tiles.
    """
    rows = list(rows)
    if strategy not in {"random", "region", "state", "temporal"}:
        raise ValueError("strategy must be random, region, state or temporal")
    if not math.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("buffer_m must be finite and nonnegative")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        parent[find(j)] = find(i)

    seen = {}
    for i, row in enumerate(rows):
        if not row.get("region_id"):
            raise ValueError("Geographic split requires region_id, including random/temporal safety grouping")
        keys = [("region_id", str(row["region_id"]))]
        if strategy == "state":
            if not row.get("state"):
                raise ValueError("State split requires state on every row")
            keys.append(("state", str(row["state"])))
        for key in ("pre", "post", "parent_id", "pair_id", "spatial_group",
                    "source_raster", "pre_source_raster", "post_source_raster",
                    "pre_raster", "post_raster", "aoi_id", "location_id"):
            if row.get(key):
                # pre/post share the same namespace to stop temporal role crossover.
                namespace = "image" if key in {"pre", "post"} else key
                if key in {"source_raster", "pre_source_raster", "post_source_raster", "pre_raster", "post_raster"}:
                    namespace = "source_raster"
                keys.append((namespace, str(row[key])))
        for key in keys:
            if key in seen:
                join(i, seen[key])
            seen[key] = i
    bounds = [_bounds(row) for row in rows]
    if any(b is not None for b in bounds):
        if any(b is None or not row.get("crs") for b, row in zip(bounds, rows)):
            raise ValueError("All rows need bounds and CRS when spatial adjacency is available")
        target = next((row["crs"] for row in rows if CRS.from_user_input(row["crs"]).is_projected
                       and abs(CRS.from_user_input(row["crs"]).axis_info[0].unit_conversion_factor - 1) < 1e-8), None)
        if target is None:
            from terradelta.utils.geo import metric_crs
            target = metric_crs(project_bounds(bounds[0], rows[0]["crs"], "EPSG:4326"))
        polygons = [box(*project_bounds(b, row["crs"], target)) for b, row in zip(bounds, rows)]
        tree = STRtree(polygons)
        for i, polygon in enumerate(polygons):
            query = polygon.buffer(buffer_m) if buffer_m else polygon
            for j in tree.query(query, predicate="intersects"):
                join(i, int(j))
    elif buffer_m:
        raise ValueError("A geographic buffer requires bounds and CRS")
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda indexes: tuple(sorted(str(rows[i].get("id", i)) for i in indexes)))


def split_manifest(rows, *, strategy="region", val_fraction=0.2, seed=0, buffer_m=0.0,
                   temporal_year=None):
    """Return train/val lists. Entire connected groups go to one side.

    Temporal validation holds out every group containing a post image acquired
    in/after temporal_year. Older images of that region are also held out. A
    single connected geography cannot produce a safe validation set: raise.
    """
    rows = list(rows)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = spatial_groups(rows, strategy=strategy, buffer_m=buffer_m)
    if len(groups) < 2:
        raise ValueError("Need at least two disconnected geographic groups for train/val")
    if strategy == "temporal":
        if isinstance(temporal_year, bool) or not isinstance(temporal_year, int):
            raise ValueError("Temporal split requires integer temporal_year")
        try:
            for row in rows:
                if int(row["year_pre"]) >= int(row["year_post"]):
                    raise ValueError("Temporal pairs require year_pre < year_post")
            validation = [g for g in groups if any(int(rows[i]["year_post"]) >= temporal_year for i in g)]
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Temporal split requires numeric year_pre < year_post") from error
    else:
        order = list(groups)
        random.Random(seed).shuffle(order)
        target, count, validation = len(rows) * val_fraction, 0, []
        for group in order[:-1]:
            if count < target:
                validation.append(group)
                count += len(group)
    val_indices = {i for group in validation for i in group}
    if not val_indices or len(val_indices) == len(rows):
        raise ValueError("Split cannot produce nonempty geographically disjoint train/val")
    return {"train": [row for i, row in enumerate(rows) if i not in val_indices],
            "val": [row for i, row in enumerate(rows) if i in val_indices]}


def assert_no_spatial_leak(train_rows, val_rows, *, buffer_m=0.0):
    train, val = list(train_rows), list(val_rows)
    for group in spatial_groups(train + val, buffer_m=buffer_m):
        if any(i < len(train) for i in group) and any(i >= len(train) for i in group):
            raise ValueError("Train/validation share a region, source pair, or neighboring footprint")
