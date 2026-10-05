"""v2.3.2 regression tests: experimental features OFF must leave V2.3.1 semantics untouched."""
import json

import numpy as np
import pytest
import torch

import terradelta.inference.v231_predictor as v231
from terradelta.inference.evidence_classifier import (
    EvidenceClassifier,
    LinearComponentClassifier,
    apply_evidence_filtering,
)
from terradelta.inference.evidence_features import (
    ABLATION_SCHEMAS,
    extract_deep_cva_and_scales,
)
from terradelta.inference.experimental import ExperimentalConfig
from terradelta.inference.local_alignment import (
    V232_ALIGNMENT_FEATURES,
    LocalAlignmentConfig,
    LocalAlignmentFeatureExtractor,
)
from terradelta.inference.pair_gate import (
    V232_PAIR_FEATURES,
    PairChangeGate,
    PairFeatureExtractor,
    PairGateConfig,
)
from terradelta.inference.v2 import V2Predictor
from terradelta.inference.v232_hooks import (
    attach_alignment_features,
    gate_filtered_row,
    validate_classifier_alignment_requirements,
)
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


def fake_predictor(monkeypatch, experimental=None, classifier_enabled=True):
    cfg = dict(CONFIG)
    if experimental is not None:
        cfg["experimental"] = experimental
    row, records = make_rows_and_records()
    monkeypatch.setattr(V2Predictor, "output_rows", lambda self, ids, img, px, pr: [dict(row)])
    monkeypatch.setattr(v231, "extract_pair_evidence_records", lambda *a, **k: [records])
    pred = object.__new__(v231.V231Predictor)
    pred.config = cfg
    pred.classifier = EvidenceClassifier(
        {"new_building": LinearComponentClassifier(["score"], [0.0], [1.0], [1.0], 0.0, threshold=0.5)},
        enabled=classifier_enabled,
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
        {"experimental": {"alignment_residual": {"enabled": True, "max_shift_image_px": 8, "feature_scales": [2]}}}
    )
    assert exp.any_enabled and exp.alignment_residual.max_shift_image_px == 8


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
    fp = PairGateConfig().fingerprint()
    model = {"feature_names": ["pair_max_candidate_score"], "mean": [0.0], "scale": [1.0],
             "weights": [1.0], "bias": 0.0, "threshold": 0.5, "expected_fingerprint": fp}
    pred, row, records = fake_predictor(monkeypatch, {"pair_gate": {"enabled": True, "model": model}})
    baseline = apply_evidence_filtering(row, records, pred.classifier, CONFIG)
    assert run(pred) == [baseline]  # score sigmoid(1)=0.73 > 0 -> logit>0 -> keep

    model["bias"] = -50.0
    pred, _, _ = fake_predictor(monkeypatch, {"pair_gate": {"enabled": True, "model": model}})
    out = run(pred)[0]
    assert out["new_building"] == "" and out["tree_removal"] == ""


def test_pair_gate_runs_when_classifier_disabled(monkeypatch):
    """Item 7: pair gate must not be silently bypassed when classifier is disabled."""
    fp = PairGateConfig().fingerprint()
    model = {"feature_names": ["pair_max_candidate_score"], "mean": [0.0], "scale": [1.0],
             "weights": [1.0], "bias": -50.0, "threshold": 0.5, "expected_fingerprint": fp}
    # classifier_enabled=False, pair_gate.enabled=True
    pred, row, records = fake_predictor(
        monkeypatch,
        experimental={"pair_gate": {"enabled": True, "model": model}},
        classifier_enabled=False,
    )
    # Must NOT silently return un-gated row; pair gate veto must apply
    out = run(pred)[0]
    assert out["new_building"] == "" and out["tree_removal"] == ""


def test_strict_alignment_feature_guard():
    """Item 6 & Phase 1: active classifier declaring alignment features requires alignment_residual.enabled=True and matching fingerprint."""
    cfg_on = LocalAlignmentConfig(enabled=True)
    align_fp = cfg_on.fingerprint()

    # 1. Ablation D (no alignment features) remains fully backward-compatible regardless of alignment config
    model_d = LinearComponentClassifier(["score"], [0.0], [1.0], [1.0], 0.0)
    clf_d = EvidenceClassifier({"new_building": model_d}, enabled=True)
    validate_classifier_alignment_requirements(clf_d, LocalAlignmentConfig(enabled=False))
    validate_classifier_alignment_requirements(clf_d, cfg_on)

    # 2. Classifier with alignment features + alignment disabled -> raises ValueError
    model_align = LinearComponentClassifier(["align_residual_reduction"], [0.0], [1.0], [1.0], 0.0)
    clf_align_nofp = EvidenceClassifier({"new_building": model_align}, enabled=True)
    cfg_off = LocalAlignmentConfig(enabled=False)
    with pytest.raises(ValueError, match="declares experimental alignment features.*enabled is False"):
        validate_classifier_alignment_requirements(clf_align_nofp, cfg_off)

    # 3. Classifier with alignment features + missing expected_alignment_fingerprint -> raises ValueError
    with pytest.raises(ValueError, match="expected_alignment_fingerprint is missing or empty"):
        validate_classifier_alignment_requirements(clf_align_nofp, cfg_on)

    # 4. Classifier with alignment features + wrong expected_alignment_fingerprint -> raises ValueError
    clf_align_bad = EvidenceClassifier({"new_building": model_align}, enabled=True, expected_alignment_fingerprint="wrong_fp_123")
    with pytest.raises(ValueError, match="expected alignment fingerprint 'wrong_fp_123' does not match"):
        validate_classifier_alignment_requirements(clf_align_bad, cfg_on)

    # 5. Classifier with alignment features + matching expected_alignment_fingerprint -> passes
    clf_align_ok = EvidenceClassifier({"new_building": model_align}, enabled=True, expected_alignment_fingerprint=align_fp)
    validate_classifier_alignment_requirements(clf_align_ok, cfg_on)

    # 6. Component model level expected_alignment_fingerprint also supported
    model_align_with_fp = LinearComponentClassifier(
        ["align_residual_reduction"], [0.0], [1.0], [1.0], 0.0, expected_alignment_fingerprint=align_fp
    )
    clf_model_fp = EvidenceClassifier({"new_building": model_align_with_fp}, enabled=True)
    validate_classifier_alignment_requirements(clf_model_fp, cfg_on)


def test_encoder_features_reused_between_cva_and_alignment():
    """Item 8: shared forward pass gives identical CVA and extracts only configured scales."""
    class FakeEncoder(torch.nn.Module):
        def forward(self, x):
            return [
                torch.zeros((1, 16, 64, 64)),
                torch.zeros((1, 32, 64, 64)),
                torch.ones((1, 64, 64, 64)) * 2.0,  # stage 2
                torch.ones((1, 128, 32, 32)) * 3.0,  # stage 3
            ]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FakeEncoder()

    model = FakeModel()
    pre = torch.zeros((1, 3, 256, 256))
    post = torch.ones((1, 3, 256, 256))

    cva_res, pre_dict, post_dict = extract_deep_cva_and_scales(
        model, pre, post, torch.device("cpu"), compute_cva=True, scales=[2, 3]
    )
    assert cva_res is not None
    cva_map, cos_map = cva_res
    assert cva_map.shape == (1, 256, 256)
    assert cos_map.shape == (1, 256, 256)
    assert set(pre_dict) == {2, 3}
    assert set(post_dict) == {2, 3}
    assert pre_dict[2].shape == (1, 64, 64, 64)
    assert pre_dict[3].shape == (1, 128, 32, 32)


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
