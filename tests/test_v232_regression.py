"""v2.3.2 regression tests: experimental features OFF must leave V2.3.1 semantics untouched."""
import json

import numpy as np
import pytest

import terradelta.inference.v231_predictor as v231
from terradelta.inference.evidence_classifier import (
    EvidenceClassifier,
    LinearComponentClassifier,
    apply_evidence_filtering,
)
from terradelta.inference.evidence_features import ABLATION_SCHEMAS
from terradelta.inference.experimental import ExperimentalConfig
from terradelta.inference.local_alignment import V232_ALIGNMENT_FEATURES, LocalAlignmentConfig
from terradelta.inference.local_alignment import LocalAlignmentFeatureExtractor
from terradelta.inference.pair_gate import (
    V232_PAIR_FEATURES,
    PairChangeGate,
    PairFeatureExtractor,
    PairGateConfig,
)
from terradelta.inference.v2 import V2Predictor
from terradelta.inference.v232_hooks import attach_alignment_features, gate_filtered_row
from terradelta.inference.v232_schema import V232_CANDIDATE_SCHEMA_D_ALIGN
from terradelta.postprocess.polygons import serialize_polygons

POLY = {"type": "Polygon", "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]}
CONFIG = {"postprocess": {"min_pos_area": 10, "classes": {"new_building": {"min_pos_area": 10}}}}
FEATURES = {
    "score": 1.0,
    "area": 100.0,
    "persistence_fraction": 1.0,
    "reverse_mean_prob": 0.1,
    "mean_probability": 0.9,
}


def make_rows_and_records():
    row = {"id": "p0", "new_building": serialize_polygons([POLY]), "tree_removal": ""}
    records = {"new_building": [{"polygon": POLY, "component_index": 0, "features": dict(FEATURES)}],
               "tree_removal": []}
    return row, records


def fake_predictor(monkeypatch, experimental=None):
    cfg = dict(CONFIG)
    if experimental is not None:
        cfg["experimental"] = experimental
    row, records = make_rows_and_records()
    monkeypatch.setattr(V2Predictor, "output_rows", lambda self, ids, img, px, pr: [dict(row)])
    monkeypatch.setattr(v231, "extract_pair_evidence_records", lambda *a, **k: [records])
    pred = object.__new__(v231.V231Predictor)
    pred.config = cfg
    pred.classifier = EvidenceClassifier(
        {"new_building": LinearComponentClassifier(["score"], [0.0], [1.0], [1.0], 0.0, threshold=0.5)}
    )
    pred.experimental = ExperimentalConfig.from_config(cfg)
    pred.pair_gate = PairChangeGate(pred.experimental.pair_gate)
    pred.pair_features = PairFeatureExtractor(pred.experimental.pair_gate)
    pred.alignment = LocalAlignmentFeatureExtractor(pred.experimental.alignment_residual)
    return pred, row, records


def run(pred):
    return pred.output_rows(["p0"], None, np.zeros((1, 2, 4, 4)), np.full((1, 2), 0.9))


def test_no_experimental_key_means_everything_off():
    exp = ExperimentalConfig.from_config({})
    assert not exp.any_enabled and not exp.pair_gate.enabled and not exp.alignment_residual.enabled
    exp = ExperimentalConfig.from_config(
        {"experimental": {"pair_gate": {"enabled": False}, "alignment_residual": {"enabled": False}}}
    )
    assert not exp.any_enabled


def test_experimental_config_validation():
    with pytest.raises(ValueError, match="Unknown experimental keys"):
        ExperimentalConfig.from_config({"experimental": {"nope": {}}})
    with pytest.raises(ValueError, match="mapping"):
        ExperimentalConfig.from_config({"experimental": [1]})
    exp = ExperimentalConfig.from_config(
        {"experimental": {"alignment_residual": {"enabled": True, "max_shift": 2, "feature_scales": [2]}}}
    )
    assert exp.any_enabled and exp.alignment_residual.max_shift == 2


def test_old_classifier_model_data_loads_unchanged():
    names = list(ABLATION_SCHEMAS["D"])
    data = {
        "version": 1,
        "enabled": True,
        "classes": {
            "new_building": {
                "model_type": "logistic_regression",
                "feature_names": names,
                "mean": [0.0] * len(names),
                "scale": [1.0] * len(names),
                "weights": [0.0] * len(names),
                "intercept": 0.0,
                "threshold": 0.5,
            }
        },
    }
    clf = EvidenceClassifier.from_dict(data)
    assert clf.models["new_building"].feature_names == tuple(names)
    assert clf.to_dict()["classes"]["new_building"]["feature_names"] == names
    assert len(names) == 13 + 22 + 12 + 7  # frozen ablation D size


def test_frozen_ablation_d_schema_untouched_and_new_schema_separate():
    assert not set(ABLATION_SCHEMAS["D"]) & set(V232_ALIGNMENT_FEATURES)
    assert V232_CANDIDATE_SCHEMA_D_ALIGN[: len(ABLATION_SCHEMAS["D"])] == ABLATION_SCHEMAS["D"]
    assert V232_CANDIDATE_SCHEMA_D_ALIGN[len(ABLATION_SCHEMAS["D"]):] == V232_ALIGNMENT_FEATURES
    assert len(set(V232_CANDIDATE_SCHEMA_D_ALIGN)) == len(V232_CANDIDATE_SCHEMA_D_ALIGN)
    for schema in ABLATION_SCHEMAS.values():
        assert not set(schema) & set(V232_PAIR_FEATURES)


def test_both_disabled_matches_frozen_v231_filtering(monkeypatch):
    pred, row, records = fake_predictor(monkeypatch)
    expected = apply_evidence_filtering(row, records, pred.classifier, CONFIG)
    assert run(pred) == [expected]
    assert "align_best_dx" not in records["new_building"][0]["features"]


def test_explicit_disabled_flags_match_absent_flags(monkeypatch):
    off, _, _ = fake_predictor(monkeypatch, {"pair_gate": {"enabled": False},
                                             "alignment_residual": {"enabled": False}})
    absent, _, _ = fake_predictor(monkeypatch)
    assert run(off) == run(absent)


def test_gate_accept_preserves_v231_output_and_reject_clears(monkeypatch):
    model = {"feature_names": ["pair_max_candidate_score"], "mean": [0.0], "scale": [1.0],
             "weights": [1.0], "bias": 0.0, "threshold": 0.5}
    pred, row, records = fake_predictor(monkeypatch, {"pair_gate": {"enabled": True, "model": model}})
    baseline = apply_evidence_filtering(row, records, pred.classifier, CONFIG)
    assert run(pred) == [baseline]  # score sigmoid(1)=0.73 > 0 -> logit>0 -> keep

    model["bias"] = -50.0
    pred, _, _ = fake_predictor(monkeypatch, {"pair_gate": {"enabled": True, "model": model}})
    out = run(pred)[0]
    assert out["new_building"] == "" and out["tree_removal"] == ""


def test_alignment_enabled_does_not_change_kept_polygons(monkeypatch):
    pred, row, records = fake_predictor(monkeypatch, {"alignment_residual": {"enabled": True}})
    monkeypatch.setattr(v231, "encoder_feature_maps",
                        lambda p, img, scales: ({s: np.zeros((1, 2, 8, 8)) for s in scales},) * 2)
    monkeypatch.setattr(v231, "candidate_masks", lambda px, cfg, name: [np.ones((256, 256), dtype=bool)])
    pred.alignment = LocalAlignmentFeatureExtractor(LocalAlignmentConfig(enabled=True, image_size=256))
    baseline = apply_evidence_filtering(row, records, pred.classifier, CONFIG)
    assert run(pred) == [baseline]
    feats = records["new_building"][0]["features"]
    assert all(name in feats for name in V232_ALIGNMENT_FEATURES)


def test_attach_alignment_guards():
    _, records = make_rows_and_records()
    ext = LocalAlignmentFeatureExtractor(
        LocalAlignmentConfig(enabled=True, image_size=16, feature_scales=(0,), bbox_padding=2, ring_radius=1)
    )
    f = np.random.default_rng(0).normal(size=(2, 16, 16)).astype(np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    with pytest.raises(ValueError, match="mask count"):
        attach_alignment_features(records, {"new_building": []}, {0: f}, {0: f}, ext)
    attach_alignment_features(records, {"new_building": [mask]}, {0: f}, {0: f}, ext)
    assert records["new_building"][0]["features"]["align_num_valid_scales"] == 1.0
    with pytest.raises(ValueError, match="collide"):
        attach_alignment_features(records, {"new_building": [mask]}, {0: f}, {0: f}, ext)


def test_gate_filtered_row_marks_cleared_class_as_not_kept():
    row, records = make_rows_and_records()
    row["new_building"] = ""  # class already emptied by min-area rules
    clf = EvidenceClassifier(
        {"new_building": LinearComponentClassifier(["score"], [0.0], [1.0], [1.0], 0.0, threshold=0.5)}
    )
    gate = PairChangeGate(PairGateConfig())
    ext = PairFeatureExtractor(PairGateConfig())
    _, result = gate_filtered_row(row, records, {"new_building": 0.9}, clf, ext, gate)
    assert result.features["building_candidate_count"] == 1.0
    assert result.features["building_kept_candidate_count"] == 0.0
    assert json.dumps(result.features)  # serializable
