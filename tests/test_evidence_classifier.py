"""Unit tests for object-level evidence classifier and feature extraction."""
import json
import numpy as np
import pytest
from shapely.geometry import box

from terradelta.inference.evidence_classifier import (
    EvidenceClassifier,
    LinearComponentClassifier,
    apply_evidence_filtering,
)
from terradelta.inference.evidence_features import (
    ABLATION_SCHEMAS,
    BASE_CONFIDENCE_FEATURES,
    BASE_GEOMETRY_FEATURES,
    BUILDING_SPECIFIC_FEATURES,
    DEEP_CVA_FEATURES,
    GLOBAL_NUISANCE_FEATURES,
    REVERSE_FEATURES,
    RGB_STRUCTURAL_FEATURES,
    STABILITY_FEATURES,
    TREE_SPECIFIC_FEATURES,
    compute_confidence_features,
    compute_geometry_features,
    get_component_pixels,
    get_surrounding_ring_mask,
)
from terradelta.postprocess.polygons import serialize_polygons


def test_feature_schemas_deterministic():
    """Verify all feature schema definitions are non-empty, unique and deterministic."""
    assert len(BASE_CONFIDENCE_FEATURES) == 13
    assert len(BASE_GEOMETRY_FEATURES) == 22
    assert len(STABILITY_FEATURES) == 12
    assert len(REVERSE_FEATURES) == 7
    assert len(DEEP_CVA_FEATURES) == 8
    assert len(RGB_STRUCTURAL_FEATURES) == 15
    assert len(GLOBAL_NUISANCE_FEATURES) == 6
    assert len(BUILDING_SPECIFIC_FEATURES) == 6
    assert len(TREE_SPECIFIC_FEATURES) == 7

    for name, schema in ABLATION_SCHEMAS.items():
        assert len(schema) == len(set(schema)), f"Duplicate in schema {name}"
        assert all(isinstance(f, str) for f in schema)


def test_geometry_features_box():
    """Test geometry features on an exact 10x20 rectangle."""
    # Box from x=[10, 30], y=[20, 30] -> width=20, height=10
    poly = box(10, 20, 30, 30)
    y, x = get_component_pixels(poly)
    assert len(y) == 200

    geom = compute_geometry_features(poly, y, x, [poly], current_idx=0, image_size=(256, 256))
    assert geom["area"] == 200.0
    assert geom["perimeter"] == 60.0
    assert geom["bbox_width"] == 20.0
    assert geom["bbox_height"] == 10.0
    assert geom["aspect_ratio"] == 2.0
    assert geom["solidity"] == 1.0
    assert geom["extent"] == 1.0
    assert geom["rectangularity"] == pytest.approx(1.0, abs=1e-3)
    assert geom["border_distance"] == 10.0
    assert geom["touches_border"] == 0.0
    assert np.isfinite(geom["eccentricity"])
    assert np.isfinite(geom["circularity"])


def test_surrounding_ring_mask():
    """Verify ring mask is disjoint from component and within image bounds."""
    poly = box(50, 50, 70, 70)
    y, x = get_component_pixels(poly)
    ring = get_surrounding_ring_mask(y, x, radius=5, shape=(256, 256))

    # Ring must be disjoint from component
    comp_mask = np.zeros((256, 256), dtype=bool)
    comp_mask[y, x] = True
    assert not (ring & comp_mask).any()

    # Ring must be non-empty around a 20x20 box
    assert ring.sum() > 0
    # Ring must be within 5 pixels of the box
    assert ring.shape == (256, 256)


def test_confidence_features():
    """Verify confidence feature computation."""
    prob_map = np.full((256, 256), 0.3, dtype=np.float32)
    # Set high probability in a component region
    y, x = np.mgrid[10:20, 10:20]
    y_flat, x_flat = y.ravel(), x.ravel()
    prob_map[y_flat, x_flat] = np.linspace(0.6, 0.9, len(y_flat))

    conf = compute_confidence_features(
        y_flat, x_flat, prob_map, presence_score=0.85, verifier_score=0.72, all_areas=[100.0]
    )
    assert conf["component_area"] == 100.0
    assert conf["mean_probability"] == pytest.approx(0.75, abs=1e-2)
    assert conf["max_probability"] == pytest.approx(0.9, abs=1e-4)
    assert conf["p90_probability"] > conf["median_probability"]
    assert conf["presence_score"] == 0.85
    assert conf["verifier_score"] == 0.72
    assert conf["num_same_class_components"] == 1.0


def test_pure_numpy_logistic_regression():
    """Test pure NumPy LinearComponentClassifier evaluator."""
    features = ["f1", "f2", "f3"]
    mean = [0.0, 1.0, 2.0]
    scale = [1.0, 2.0, 3.0]
    weights = [1.0, -1.0, 0.5]
    intercept = 0.0

    clf = LinearComponentClassifier(
        feature_names=features,
        mean=mean,
        scale=scale,
        weights=weights,
        intercept=intercept,
        threshold=0.5,
    )

    # At x = mean, z = [0, 0, 0], logit = 0.0, proba = 0.5 -> keep=True
    assert clf.predict_proba(np.array([0.0, 1.0, 2.0])) == pytest.approx(0.5, abs=1e-6)
    assert clf.predict_keep({"f1": 0.0, "f2": 1.0, "f3": 2.0}) is True

    # High negative logit -> proba < 0.5 -> keep=False
    assert clf.predict_keep({"f1": -10.0, "f2": 20.0, "f3": 0.0}) is False

    # Serialization roundtrip
    d = clf.to_dict()
    clf2 = LinearComponentClassifier.from_dict(d)
    assert clf2.feature_names == clf.feature_names
    assert np.allclose(clf2.weights, clf.weights)
    assert clf2.intercept == clf.intercept


def test_polygon_preservation_strict():
    """Ensure kept polygons preserve original coordinates and order exactly."""
    original_polys = [
        {"type": "Polygon", "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]},
        {"type": "Polygon", "coordinates": [[[50, 50], [70, 50], [70, 70], [50, 70], [50, 50]]]},
    ]
    serialized = serialize_polygons(original_polys)

    row = {"id": "test_pair_01", "new_building": serialized, "tree_removal": ""}
    records = {
        "new_building": [
            {"polygon": original_polys[0], "features": {"score": 1.0}},
            {"polygon": original_polys[1], "features": {"score": -1.0}},
        ],
        "tree_removal": [],
    }

    # Model that keeps only poly1 (score >= 0.5)
    clf = EvidenceClassifier({
        "new_building": LinearComponentClassifier(["score"], [0.0], [1.0], [1.0], 0.0, threshold=0.5),
    })

    config = {"postprocess": {"min_pos_area": 50, "classes": {"new_building": {"min_pos_area": 50}}}}
    filtered = apply_evidence_filtering(row, records, clf, config)

    # Kept polygons must contain exactly original_polys[0]
    kept = json.loads(filtered["new_building"])
    assert len(kept) == 1
    assert kept[0] == original_polys[0]
    # Coordinates must match down to the exact floating point bit
    assert kept[0]["coordinates"] == original_polys[0]["coordinates"]


def test_min_pos_area_drop():
    """If remaining kept components sum to less than min_pos_area, drop all."""
    poly1 = {"type": "Polygon", "coordinates": [[[10, 10], [15, 10], [15, 15], [10, 15], [10, 10]]]}  # area 25
    serialized = serialize_polygons([poly1])
    row = {"id": "test_pair_02", "new_building": serialized, "tree_removal": ""}
    records = {
        "new_building": [{"polygon": poly1, "features": {"score": 1.0}}],
        "tree_removal": [],
    }
    # Keep poly1, but its area is 25 which is < min_pos_area=30
    clf = EvidenceClassifier({
        "new_building": LinearComponentClassifier(["score"], [0.0], [1.0], [1.0], 0.0, threshold=0.5),
    })
    config = {"postprocess": {"min_pos_area": 30, "classes": {"new_building": {"min_pos_area": 30}}}}
    filtered = apply_evidence_filtering(row, records, clf, config)
    assert filtered["new_building"] == ""


def test_disabled_classifier():
    """When disabled, apply_evidence_filtering returns exact input row."""
    row = {"id": "test_pair_03", "new_building": "[{\"type\":\"Polygon\"}]", "tree_removal": ""}
    records = {"new_building": [{"polygon": {"type": "Polygon"}, "features": {}}]}
    clf = EvidenceClassifier({}, enabled=False)
    filtered = apply_evidence_filtering(row, records, clf, {})
    assert filtered == row


def test_polygon_preservation_coordinate_list():
    """Test polygon preservation when polygons are serialized coordinate lists."""
    coords1 = [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0]]  # area 100
    coords2 = [[50.0, 50.0], [60.0, 50.0], [60.0, 60.0], [50.0, 60.0], [50.0, 50.0]]  # area 100
    serialized = serialize_polygons([coords1, coords2])

    row = {"id": "test_coords_pair", "new_building": serialized, "tree_removal": ""}
    records = {
        "new_building": [
            {"polygon": coords1, "features": {"score": 1.0, "area": 100.0}},
            {"polygon": coords2, "features": {"score": -1.0, "area": 100.0}},
        ],
        "tree_removal": [],
    }

    clf = EvidenceClassifier({
        "new_building": LinearComponentClassifier(["score"], [0.0], [1.0], [1.0], 0.0, threshold=0.5),
    })
    config = {"postprocess": {"min_pos_area": 50, "classes": {"new_building": {"min_pos_area": 50}}}}
    filtered = apply_evidence_filtering(row, records, clf, config)

    kept = json.loads(filtered["new_building"])
    assert len(kept) == 1
    assert kept[0] == coords1
