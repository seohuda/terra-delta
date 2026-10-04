import json

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from terradelta.postprocess import (
    class_config, fill_holes, filter_components, mask_to_polygons_reference,
    postprocess_mask, predictions_to_polygons, probabilities_to_masks,
)


def test_default_argmax_matches_original_for_both_classes():
    rng = np.random.default_rng(51)
    logits = rng.random((3, 20, 24))
    p = logits / logits.sum(axis=0)
    result = predictions_to_polygons(p, {})
    labels = p.argmax(axis=0)
    for k, name in enumerate(("new_building", "tree_removal"), 1):
        expected = mask_to_polygons_reference(labels == k)
        assert result[name] == (json.loads(expected) if expected else [])


def test_parent_inference_import_and_direct_postprocess_config():
    from terradelta.postprocess.polygons import predictions_to_polygons as adapter, serialize_polygons
    p = np.zeros((3, 8, 8))
    p[1] = 1
    config = {"mode": "threshold", "classes": {"new_building": {"threshold": .6}}}
    expected = predictions_to_polygons(p, {"postprocess": config})
    assert adapter(p, config) == expected
    assert json.loads(serialize_polygons(expected["new_building"])) == expected["new_building"]


def test_threshold_classes_can_overlap_and_have_independent_settings():
    p = np.zeros((3, 12, 12), np.float32)
    p[0] = .1
    p[1] = .45
    p[2] = .45
    config = {"postprocess": {"mode": "threshold", "backend": "runs",
              "new_building": {"threshold": .4}, "tree_removal": {"threshold": .4}}}
    result = predictions_to_polygons(p, config)
    geometries = [unary_union([Polygon(polygon) for polygon in result[name]])
                  for name in ("new_building", "tree_removal")]
    assert geometries[0].intersection(geometries[1]).area == 144
    config["postprocess"]["tree_removal"]["threshold"] = .5
    result = predictions_to_polygons(p, config)
    assert result["new_building"] and result["tree_removal"] == []


def test_threshold_inclusive_and_independent_morphology():
    p = np.zeros((3, 9, 9))
    p[0] = 1
    p[:, 4, 4] = [.1, .45, .45]
    config = {"mode": "threshold", "min_area": 0, "min_pos_area": 0,
              "classes": {"new_building": {"threshold": .45, "dilation": 1},
                          "tree_removal": {"threshold": .45}}}
    masks = probabilities_to_masks(p, config)
    assert masks["new_building"].sum() == 9
    assert masks["tree_removal"].sum() == 1


def test_class_overrides_and_nested_morphology_merge_without_mutation():
    config = {"postprocess": {"min_area": 30, "building_threshold": .6,
              "morphology": {"opening": 1, "closing": 0},
              "classes": {"new_building": {"min_component_area": 40,
                                           "morphology": {"closing": 1}}},
              "new_building": {"morphology": {"dilation": 2}}}}
    before = json.dumps(config)
    building = class_config(config, "new_building")
    tree = class_config(config, "tree_removal")
    assert building["min_area"] == 40
    assert building["threshold"] == .6
    assert building["morphology"] == {"opening": 1, "closing": 1, "dilation": 2}
    assert tree["min_area"] == 30 and tree["threshold"] == .5
    assert json.dumps(config) == before


def test_class_canonical_overrides_common_alias_and_direct_morphology():
    config = {"min_component_area": 100, "morphology": {"opening": 1},
              "classes": {"new_building": {"min_area": 30, "opening": 0}}}
    options = class_config(config, "new_building")
    assert options["min_area"] == 30
    assert options["morphology"]["opening"] == 0
    assert "min_component_area" not in options


def test_default_morphology_copies_without_change():
    m = np.eye(6, dtype=bool)
    result = postprocess_mask(m)
    assert np.array_equal(result, m)
    result[:] = False
    assert m.any()


def test_opening_removes_speckle_and_preserves_large_square():
    m = np.zeros((12, 12), bool)
    m[3:8, 3:8] = True
    m[10, 10] = True
    result = postprocess_mask(m, {"opening": 1})
    assert result.sum() == 25 and result[3:8, 3:8].all()


def test_closing_fills_internal_pixel_hole():
    m = np.zeros((12, 12), bool)
    m[3:9, 3:9] = True
    m[5, 5] = False
    result = postprocess_mask(m, {"closing": 1})
    assert result.sum() == 36 and result[5, 5]


def test_dilation_and_erosion_use_square_pixel_radius_and_zero_border():
    m = np.zeros((8, 8), bool)
    m[0, 0] = True
    assert postprocess_mask(m, {"dilation": 1}).sum() == 4
    m[:] = True
    result = postprocess_mask(m, {"erosion": 1})
    assert result.sum() == 36
    assert not result[0].any() and not result[:, -1].any()


def test_hole_filling_preserves_border_connected_background():
    m = np.ones((10, 10), bool)
    m[3:6, 3:6] = False
    m[:4, 8] = False
    result = fill_holes(m)
    assert result[3:6, 3:6].all()
    assert not result[:4, 8].any()
    assert np.array_equal(result, postprocess_mask(m, {"fill_holes": True}))


def test_connected_components_four_vs_eight():
    m = np.eye(5, dtype=bool)
    assert not filter_components(m, min_area=2, connectivity=4).any()
    assert np.array_equal(filter_components(m, min_area=5, connectivity=8), m)
    assert not filter_components(m, min_area=6, connectivity=8).any()


def test_cc_filter_after_morphology():
    m = np.zeros((12, 12), bool)
    m[5, 5] = True
    assert postprocess_mask(m, {"dilation": 1, "cc_min_area": 9}).sum() == 9
    assert postprocess_mask(m, {"dilation": 1, "cc_min_area": 10}).sum() == 0


@pytest.mark.parametrize("p", [np.zeros((2, 8, 8)), np.zeros((3, 0, 8)),
                             np.full((3, 8, 8), np.nan), np.full((3, 8, 8), 1.01),
                             np.full((3, 8, 8), -.01)])
def test_invalid_probabilities_fail(p):
    with pytest.raises(ValueError):
        predictions_to_polygons(p)


@pytest.mark.parametrize("config", [{"mode": "invalid"}, {"mode": "threshold", "threshold": -1},
                                    {"opening": -1}, {"closing": 1.5}, {"dilation": True},
                                    {"connectivity": 6}, {"cc_min_area": -1}])
def test_invalid_settings_fail(config):
    with pytest.raises(ValueError):
        predictions_to_polygons(np.zeros((3, 8, 8)), config)


def test_no_rasterio_or_scipy_needed_for_default_or_morphology(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("rasterio", "scipy"):
            raise AssertionError(f"unexpected dependency {name}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    p = np.zeros((3, 10, 10))
    p[1, 2:8, 2:8] = 1
    assert predictions_to_polygons(p)["new_building"]
    assert predictions_to_polygons(p, {"backend": "runs", "opening": 1})["new_building"]
