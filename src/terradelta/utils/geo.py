"""Explicit CRS conversion, footprint rasterization and bounded raster reads."""

from __future__ import annotations

import math

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom


def validate_bounds(bounds, *, geographic=False):
    if len(bounds) != 4 or not all(math.isfinite(float(v)) for v in bounds):
        raise ValueError("Bounds must be four finite coordinates: west,south,east,north")
    west, south, east, north = map(float, bounds)
    if west >= east or south >= north:
        raise ValueError("Bounds must be increasing (split dateline AOIs explicitly)")
    if geographic and (west < -180 or east > 180 or south < -90 or north > 90):
        raise ValueError("WGS84 bounds are out of range")
    return west, south, east, north


def project_bounds(bounds, src_crs, dst_crs):
    bounds = validate_bounds(bounds)
    return transform_bounds(src_crs, dst_crs, *bounds, densify_pts=21)


def transform_geometry(geometry, src_crs, dst_crs):
    """Reproject GeoJSON geometry, including interior rings and MultiPolygons."""
    if not src_crs or not dst_crs:
        raise ValueError("Both geometry CRS and raster CRS must be explicit")
    return transform_geom(src_crs, dst_crs, geometry, precision=-1)


def pixel_coordinates(coordinates, transform, *, src_crs=None, raster_crs=None):
    """World coordinates -> floating-point pixel edge coordinates (col,row)."""
    affine = transform if isinstance(transform, Affine) else Affine(*transform[:6])
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Coordinates must have shape N,2")
    if bool(src_crs) != bool(raster_crs):
        raise ValueError("Supply both CRS arguments or neither")
    if src_crs and CRS.from_user_input(src_crs) != CRS.from_user_input(raster_crs):
        transformer = Transformer.from_crs(src_crs, raster_crs, always_xy=True)
        xs, ys = transformer.transform(points[:, 0], points[:, 1])
        points = np.column_stack((xs, ys))
    inverse = ~affine
    return np.array([inverse * tuple(point) for point in points])


def rasterize_geometries(geometries, *, out_shape, transform, src_crs, dst_crs,
                         all_touched=False, max_pixels=16_777_216):
    """Binary mask; holes stay background. Empty geometry list gives zero mask."""
    height, width = map(int, out_shape)
    if height <= 0 or width <= 0 or height * width > max_pixels:
        raise ValueError("Rasterization exceeds pixel limit or has invalid shape")
    affine = transform if isinstance(transform, Affine) else Affine(*transform[:6])
    projected = []
    for geometry in geometries:
        geometry = geometry.__geo_interface__ if hasattr(geometry, "__geo_interface__") else geometry
        geometry = geometry.get("geometry", geometry)
        if geometry is None:
            continue
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Building footprints must be Polygon or MultiPolygon")
        projected.append((transform_geometry(geometry, src_crs, dst_crs), 1))
    if not projected:
        return np.zeros((height, width), dtype=np.uint8)
    return rasterize(projected, out_shape=(height, width), transform=affine,
                     fill=0, dtype="uint8", all_touched=all_touched)


def metric_crs(bounds_wgs84):
    """Select a local UTM CRS for a small AOI; explicit CRS is preferable."""
    west, south, east, north = validate_bounds(bounds_wgs84, geographic=True)
    lon, lat = (west + east) / 2, (south + north) / 2
    if not -80 <= lat <= 84:
        raise ValueError("Select an explicit polar projected CRS")
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
