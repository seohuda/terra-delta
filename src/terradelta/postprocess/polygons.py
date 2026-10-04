"""Pixel-corner polygon conversion; exported exteriors deliberately have no holes.

The reference body follows the preserved official notebook. The optimized run
backend coalesces identical runs in consecutive rows before the same GEOS union.
It preserves geometry, but vertex/order serialization can differ from reference.
"""

import json
import math
from numbers import Integral

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .reference import mask_to_polygons as _official_mask_to_polygons


def _validate_parameters(min_area, min_pos_area, simplify_px, ndigits):
    for name, value in (("min_area", min_area), ("min_pos_area", min_pos_area),
                        ("simplify_px", simplify_px)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if isinstance(ndigits, bool) or not isinstance(ndigits, Integral) or ndigits < 0:
        raise ValueError("ndigits must be a nonnegative integer")


def _binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    if not np.isfinite(array).all():
        raise ValueError("mask must contain finite values")
    return array.astype(bool)


def mask_to_polygons_reference(mask, min_area=30, min_pos_area=20,
                               simplify_px=.5, ndigits=2):
    """Original baseline JSON string (or ``''``), with parameterized constants.

    For valid 2-D masks, the conversion body and ordering are preserved verbatim
    from ``baseline/original/03_illegal structure submission/predict.ipynb``.
    Filtering uses polygon area *before* simplify and removal of interior rings.
    """
    _validate_parameters(min_area, min_pos_area, simplify_px, ndigits)
    _binary_mask(mask)
    if (min_area, min_pos_area, simplify_px, ndigits) == (30, 20, .5, 2):
        return _official_mask_to_polygons(mask)
    MIN_AREA, MIN_POS_AREA = min_area, min_pos_area
    SIMPLIFY_PX, NDIGITS = simplify_px, ndigits
    return serialize_polygons(reference_exteriors(reference_components(mask), MIN_AREA, MIN_POS_AREA,
                                                  SIMPLIFY_PX, NDIGITS))


def reference_components(mask):
    """Exact reference union before filtering; reusable for cached-map search."""
    m = _binary_mask(mask)
    boxes = []
    for r in np.flatnonzero(m.any(axis=1)):
        pad = np.concatenate(([0], m[r].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(pad))
        for s, e in zip(edges[::2], edges[1::2]):
            boxes.append(box(float(s), float(r), float(e), float(r + 1)))
    if not boxes:
        return []
    g = unary_union(boxes)
    parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    return [p for p in parts if isinstance(p, Polygon)]


def reference_exteriors(parts, min_area=30, min_pos_area=20, simplify_px=.5, ndigits=2):
    """Apply original pre-export filters and simplification to cached components."""
    _validate_parameters(min_area, min_pos_area, simplify_px, ndigits)
    parts = [p for p in parts if p.area >= min_area]
    if not parts or sum(p.area for p in parts) < min_pos_area:
        return []
    out = []
    for p in parts:
        p = p.simplify(simplify_px, preserve_topology=True)
        if p.is_empty or p.area <= 0:
            continue
        out.append([[round(float(x), ndigits), round(float(y), ndigits)] for x, y in list(p.exterior.coords)[:-1]])
    return out


def _run_rectangles(mask):
    """Coalesce equal horizontal runs vertically, without approximate contours."""
    active = {}
    rectangles = []
    for row in range(mask.shape[0] + 1):
        if row < mask.shape[0]:
            padded = np.concatenate(([0], mask[row].astype(np.int8), [0]))
            edges = np.flatnonzero(np.diff(padded))
            runs = {(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])}
        else:
            runs = set()
        for run in sorted(active.keys() - runs):
            start_row = active.pop(run)
            rectangles.append(box(run[0], start_row, run[1], row))
        for run in sorted(runs - active.keys()):
            active[run] = row
    return rectangles


def _parts_from_geometry(geometry):
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return []


def mask_to_polygons(mask, min_area=30, min_pos_area=20, simplify_px=.5,
                     ndigits=2, *, backend="runs", max_polygons=None):
    """Return unclosed exterior lists, in pixel-corner coordinates.

    ``runs`` requires only NumPy and Shapely. ``rasterio`` is an explicit lazy
    optional backend using connectivity 4 and identity pixel coordinates.
    ``reference`` reproduces exact baseline output when no polygon cap is used.
    A cap keeps the largest polygons (stable bounds tie-break), before total-area
    filtering. No holes can be serialized; interiors are filled on export. Runs
    and rasterio paths also recheck rounded exterior area and emitted union area;
    the reference path intentionally retains the original pre-export filters.
    """
    _validate_parameters(min_area, min_pos_area, simplify_px, ndigits)
    m = _binary_mask(mask)
    if backend == "optimized":
        backend = "runs"
    if max_polygons is not None and (isinstance(max_polygons, bool)
            or not isinstance(max_polygons, Integral) or max_polygons < 0):
        raise ValueError("max_polygons must be a nonnegative integer or None")
    if backend == "reference" and max_polygons is None:
        cell = mask_to_polygons_reference(m, min_area, min_pos_area, simplify_px, ndigits)
        return json.loads(cell) if cell else []
    if backend not in {"runs", "reference", "rasterio"}:
        raise ValueError("backend must be 'reference', 'runs', or 'rasterio'")
    if not m.any() or max_polygons == 0:
        return []
    if backend == "rasterio":
        from rasterio.features import shapes
        from shapely.geometry import shape
        geometry = unary_union([shape(g) for g, value in shapes(
            m.astype(np.uint8), mask=m, connectivity=4) if value == 1])
    else:
        geometry = unary_union(_run_rectangles(m))
    parts = [p for p in _parts_from_geometry(geometry) if p.area >= min_area]
    if max_polygons is not None:
        parts = sorted(parts, key=lambda p: (-p.area, p.bounds))[:max_polygons]
    if not parts or sum(p.area for p in parts) < min_pos_area:
        return []
    result = []
    for polygon in parts:
        simplified = polygon.simplify(simplify_px, preserve_topology=True)
        if simplified.is_empty or simplified.area <= 0:
            continue
        exterior = [[round(float(x), ndigits), round(float(y), ndigits)]
                    for x, y in list(simplified.exterior.coords)[:-1]]
        # Rounding can collapse tiny polygons or introduce a crossing. Optimized
        # mode drops those instead of exporting an invalid submission cell.
        rounded = Polygon(exterior)
        if rounded.is_valid and rounded.area > 0 and rounded.area >= min_area:
            result.append(exterior)
    if result and unary_union([Polygon(exterior) for exterior in result]).area < min_pos_area:
        return []
    return result


def serialize_polygons(polygons):
    """Compact JSON cell, or the baseline empty string for no exteriors."""
    return json.dumps(polygons, separators=(",", ":"), allow_nan=False) if len(polygons) else ""


mask_to_polygons_optimized = mask_to_polygons


def predictions_to_polygons(probabilities, config=None):
    """List adapter exposed here for inference's established import contract."""
    from .thresholds import predictions_to_polygons as convert
    return convert(probabilities, config)
