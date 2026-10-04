"""Geometry and byte-for-byte reference checks; no model or optimizer is used."""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from terradelta.postprocess import (
    mask_to_polygons, mask_to_polygons_reference, serialize_polygons,
)
from terradelta.postprocess import reference
from terradelta.postprocess.polygons import _run_rectangles


def exported_geometry(polygons):
    return unary_union([Polygon(exterior) for exterior in polygons])


@pytest.fixture
def notebook_reference():
    path = Path(__file__).resolve().parents[1] / "baseline/original/03_illegal structure submission/predict.ipynb"
    if not path.exists():
        pytest.skip("official notebook is not available in this checkout")
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "def mask_to_polygons(" in "".join(cell.get("source", [])))
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "mask_to_polygons")
    namespace = {"np": np, "json": json, "box": box, "Polygon": Polygon,
                 "MultiPolygon": MultiPolygon, "unary_union": unary_union,
                 "MIN_AREA": 30, "MIN_POS_AREA": 20., "SIMPLIFY_PX": .5, "NDIGITS": 2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["mask_to_polygons"], ast.get_source_segment(source, node)


def cases():
    empty = np.zeros((12, 14), bool)
    pixel = empty.copy()
    pixel[1, 2] = True
    small = empty.copy()
    small[1:4, 1:6] = True
    disconnected = empty.copy()
    disconnected[0:6, 0:5] = True
    disconnected[6:12, 9:14] = True
    border = empty.copy()
    border[:8, :6] = True
    donut = np.ones((12, 14), bool)
    donut[3:9, 3:11] = False
    line = np.zeros((4, 40), bool)
    line[1] = True
    full = np.ones((12, 14), bool)
    diagonal = np.eye(12, dtype=bool)
    return [empty, pixel, small, disconnected, border, donut, line, full, diagonal]


def test_verbatim_reference_source(notebook_reference):
    _, source = notebook_reference
    assert inspect.getsource(reference.mask_to_polygons).strip() == source.strip()
    assert (reference.MIN_AREA, reference.MIN_POS_AREA, reference.SIMPLIFY_PX, reference.NDIGITS) == (30, 20., .5, 2)


@pytest.mark.parametrize("mask", cases(), ids=["empty", "pixel", "under20", "disconnected", "border", "donut", "line", "full", "diagonal"])
def test_reference_is_exact_notebook(mask, notebook_reference):
    original, _ = notebook_reference
    expected = original(mask)
    assert reference.mask_to_polygons(mask) == expected
    assert mask_to_polygons_reference(mask) == expected
    assert serialize_polygons(mask_to_polygons(mask, backend="reference")) == expected


def test_random_reference_regression(notebook_reference):
    original, _ = notebook_reference
    rng = np.random.default_rng(47)
    for _ in range(20):
        mask = rng.random((16, 21)) > .28
        assert mask_to_polygons_reference(mask) == original(mask)


@pytest.mark.parametrize("mask", cases())
def test_run_geometry_equals_reference_pixel_union(mask):
    ref = mask_to_polygons_reference(mask, min_area=0, min_pos_area=0, simplify_px=0)
    expected = exported_geometry(json.loads(ref) if ref else [])
    optimized = exported_geometry(mask_to_polygons(mask, min_area=0, min_pos_area=0, simplify_px=0))
    assert optimized.equals(expected)


def test_random_optimized_geometry_preserves_pixel_boxes():
    rng = np.random.default_rng(43)
    for _ in range(20):
        mask = rng.random((12, 15)) > .6
        rectangles = _run_rectangles(mask)
        geometry = unary_union(rectangles)
        pixel_boxes = unary_union([box(x, y, x + 1, y + 1) for y, x in zip(*np.nonzero(mask))])
        assert geometry.equals(pixel_boxes)
        assert geometry.area == int(mask.sum())


def test_full_rectangle_coalesces_runs():
    rectangles = _run_rectangles(np.ones((256, 256), bool))
    assert len(rectangles) == 1
    assert rectangles[0].bounds == (0, 0, 256, 256)


def test_optimized_backend_name_used_by_inference_config():
    m = np.ones((8, 8), bool)
    assert mask_to_polygons(m, backend="optimized") == mask_to_polygons(m, backend="runs")


def test_pixel_corners_not_pixel_centers():
    m = np.zeros((8, 8), bool)
    m[2:7, 1:7] = True
    polygons = mask_to_polygons(m, simplify_px=0)
    assert exported_geometry(polygons).bounds == (1, 2, 7, 7)
    assert exported_geometry(polygons).area == 30
    assert polygons[0][0] != polygons[0][-1]


def test_minimum_component_area_precedes_total_area():
    m = np.zeros((12, 12), bool)
    m[:4, :5] = True
    m[8:12, 7:12] = True
    assert mask_to_polygons(m) == []  # total 40, but each component <30
    assert len(mask_to_polygons(m, min_area=20)) == 2
    assert mask_to_polygons(m, min_area=20, min_pos_area=41) == []


def test_exact_area_threshold_is_inclusive():
    assert exported_geometry(mask_to_polygons(np.ones((5, 6)), simplify_px=0)).area == 30
    assert mask_to_polygons(np.ones((4, 5)), min_area=0, min_pos_area=20)
    assert mask_to_polygons(np.ones((1, 19)), min_area=0, min_pos_area=20) == []


def test_holes_filled_only_on_export_and_filter_uses_original_area():
    m = np.ones((10, 10), bool)
    m[1:9, 1:9] = False
    polygons = mask_to_polygons(m, min_area=30, simplify_px=0)
    assert m.sum() == 36
    assert exported_geometry(polygons).area == 100
    assert mask_to_polygons(m, min_area=37, simplify_px=0) == []
    assert mask_to_polygons(m, min_area=0, min_pos_area=37, simplify_px=0) == []


def test_polygon_cap_uses_largest_and_checks_total_after_cap():
    m = np.zeros((20, 20), bool)
    m[:5, :6] = True
    m[10:16, 10:17] = True
    polygons = mask_to_polygons(m, max_polygons=1, simplify_px=0)
    assert exported_geometry(polygons).area == 42
    assert mask_to_polygons(m, max_polygons=1, min_pos_area=43) == []
    assert mask_to_polygons(m, max_polygons=0) == []


def test_optimized_rechecks_emitted_area_after_simplification():
    m = np.zeros((8, 8), bool)
    m[1:6, 1:7] = True
    m[6, 3] = True  # raw area31; tolerance1 removes this protrusion -> area30
    assert mask_to_polygons(m, min_area=0, min_pos_area=31, simplify_px=1) == []
    assert mask_to_polygons(m, min_area=31, min_pos_area=0, simplify_px=1) == []
    reference_polygons = json.loads(mask_to_polygons_reference(
        m, min_area=31, min_pos_area=31, simplify_px=1))
    assert exported_geometry(reference_polygons).area == 30
    assert exported_geometry(mask_to_polygons(m, min_area=30, min_pos_area=30, simplify_px=1)).area == 30


def test_optional_rasterio_has_exact_corner_geometry():
    pytest.importorskip("rasterio")
    m = cases()[5]
    result = mask_to_polygons(m, backend="rasterio", simplify_px=0)
    expected = mask_to_polygons(m, backend="runs", simplify_px=0)
    assert exported_geometry(result).equals(exported_geometry(expected))


@pytest.mark.parametrize("kwargs", [{"min_area": -1}, {"min_pos_area": float("nan")},
                                    {"simplify_px": float("inf")}, {"ndigits": -1},
                                    {"ndigits": 1.2}, {"max_polygons": -1},
                                    {"backend": "contours"}])
def test_invalid_polygon_options_fail(kwargs):
    with pytest.raises(ValueError):
        mask_to_polygons(np.ones((8, 8)), **kwargs)


def test_invalid_masks_and_serialization_fail():
    with pytest.raises(ValueError):
        mask_to_polygons(np.ones((3, 4, 5)))
    with pytest.raises(ValueError):
        mask_to_polygons(np.array([[np.nan]]))
    with pytest.raises(ValueError):
        serialize_polygons([[[float("nan"), 0], [1, 0], [1, 1]]])
    assert serialize_polygons([]) == ""
