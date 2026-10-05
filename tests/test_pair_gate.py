"""Unit tests for the v2.3.2 pair-level change gate (synthetic data only)."""
import math

import numpy as np
import pytest

from terradelta.inference.pair_gate import (
    CROSS_CLASS_FEATURES,
    PAIR_GATE_CLASSES,
    PAIR_GATE_VERSION,
    SCOPE_STEMS,
    SCOPES,
    V232_PAIR_FEATURES,
    LinearPairGateModel,
    PairCandidate,
    PairChangeGate,
    PairFeatureExtractor,
    PairGateConfig,
    PairTrainingContract,
    apply_pair_gate,
)
from terradelta.inference.v2 import CLASSES


def cand(score=0.8, kept=True, area=100.0, stab=1.0, rev=0.1, conf=0.9):
    return PairCandidate(score, kept, area, stab, rev, conf)


def extractor(**kw):
    return PairFeatureExtractor(PairGateConfig(**kw))


def model_dict(**over):
    data = {
        "feature_names": ["pair_max_candidate_score", "pair_candidate_count"],
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "weights": [2.0, 0.0],
        "bias": -1.0,
        "threshold": 0.5,
    }
    data.update(over)
    return data


def test_classes_match_production_classes():
    assert PAIR_GATE_CLASSES == CLASSES


def test_schema_is_deterministic_unique_and_ordered():
    assert len(V232_PAIR_FEATURES) == len(set(V232_PAIR_FEATURES))
    assert len(V232_PAIR_FEATURES) == len(SCOPES) * len(SCOPE_STEMS) + len(CROSS_CLASS_FEATURES)
    assert V232_PAIR_FEATURES[0] == "pair_candidate_count"
    assert V232_PAIR_FEATURES[-1] == "building_vs_tree_score_margin"
    feats = extractor().extract({"new_building": [cand()]}, {})
    assert tuple(feats) == V232_PAIR_FEATURES


def test_empty_candidate_set():
    feats = extractor().extract({}, {})
    assert feats["pair_candidate_count"] == 0.0
    assert feats["pair_max_candidate_score"] == 0.0
    assert feats["both_classes_present"] == 0.0
    assert all(math.isfinite(v) for v in feats.values())


def test_one_candidate():
    feats = extractor().extract({"new_building": [cand(score=0.7, area=256.0)]}, {"new_building": 0.9})
    assert feats["building_candidate_count"] == 1.0
    assert feats["building_second_candidate_score"] == 0.0
    assert feats["building_score_margin_top1_top2"] == pytest.approx(0.7)
    assert feats["building_fraction_of_image_covered"] == pytest.approx(256.0 / 65536)
    assert feats["tree_candidate_count"] == 0.0
    assert feats["building_presence_probability"] == pytest.approx(0.9)


def test_multiple_candidates_and_kept_flags():
    cands = [cand(0.9, True, 100), cand(0.5, False, 50), cand(0.7, True, 30, conf=0.2)]
    feats = extractor(top_k=2).extract({"tree_removal": cands}, {})
    assert feats["tree_candidate_count"] == 3.0
    assert feats["tree_kept_candidate_count"] == 2.0
    assert feats["tree_max_candidate_score"] == 0.9
    assert feats["tree_second_candidate_score"] == 0.7
    assert feats["tree_mean_top_k_score"] == pytest.approx(0.8)
    assert feats["tree_total_candidate_area"] == 180.0
    assert feats["tree_total_kept_area"] == 130.0
    assert feats["tree_num_high_confidence_candidates"] == 2.0


def test_class_separation_and_cross_class():
    feats = extractor().extract(
        {"new_building": [cand(0.9, area=10)], "tree_removal": [cand(0.4, area=20), cand(0.3, area=5)]}, {}
    )
    assert feats["building_candidate_count"] == 1.0
    assert feats["tree_candidate_count"] == 2.0
    assert feats["pair_candidate_count"] == 3.0
    assert feats["total_candidates_all_classes"] == 3.0
    assert feats["total_area_all_classes"] == 35.0
    assert feats["max_score_any_class"] == 0.9
    assert feats["both_classes_present"] == 1.0
    assert feats["building_vs_tree_score_margin"] == pytest.approx(0.5)


def test_strict_pair_candidate_validation():
    """Item 9: PairCandidate must reject invalid probabilities and negative areas."""
    # Out of [0, 1] bounds
    with pytest.raises(ValueError, match="score"):
        PairCandidate(score=-0.1, kept=True, area=10.0, tta_stability=1.0, reverse_support=0.5, confidence=0.5)
    with pytest.raises(ValueError, match="score"):
        PairCandidate(score=1.1, kept=True, area=10.0, tta_stability=1.0, reverse_support=0.5, confidence=0.5)
    with pytest.raises(ValueError, match="confidence"):
        PairCandidate(score=0.5, kept=True, area=10.0, tta_stability=1.0, reverse_support=0.5, confidence=-0.01)
    with pytest.raises(ValueError, match="confidence"):
        PairCandidate(score=0.5, kept=True, area=10.0, tta_stability=1.0, reverse_support=0.5, confidence=1.5)
    with pytest.raises(ValueError, match="tta_stability"):
        PairCandidate(score=0.5, kept=True, area=10.0, tta_stability=-0.1, reverse_support=0.5, confidence=0.5)
    with pytest.raises(ValueError, match="reverse_support"):
        PairCandidate(score=0.5, kept=True, area=10.0, tta_stability=0.5, reverse_support=1.2, confidence=0.5)

    # Negative area or non-finite area
    with pytest.raises(ValueError, match="area"):
        PairCandidate(score=0.5, kept=True, area=-1.0, tta_stability=0.5, reverse_support=0.5, confidence=0.5)
    with pytest.raises(ValueError, match="area"):
        PairCandidate(score=0.5, kept=True, area=float("nan"), tta_stability=0.5, reverse_support=0.5, confidence=0.5)
    with pytest.raises(ValueError, match="area"):
        PairCandidate(score=0.5, kept=True, area=float("inf"), tta_stability=0.5, reverse_support=0.5, confidence=0.5)

    # kept must be boolean
    with pytest.raises(ValueError, match="kept"):
        PairCandidate(score=0.5, kept="yes", area=10.0, tta_stability=0.5, reverse_support=0.5, confidence=0.5)


def test_strict_pair_presence_validation():
    """Item 9: PairFeatureExtractor must reject invalid presence probabilities."""
    with pytest.raises(ValueError, match="presence probability"):
        extractor().extract({}, {"new_building": -0.1})
    with pytest.raises(ValueError, match="presence probability"):
        extractor().extract({}, {"new_building": 1.2})
    with pytest.raises(ValueError, match="presence probability"):
        extractor().extract({}, {"new_building": float("nan")})


def test_unknown_class_and_nonfinite_candidate_rejected():
    with pytest.raises(ValueError, match="Unknown classes"):
        extractor().extract({"road": [cand()]}, {})


def test_model_validation_errors():
    for bad, msg in [
        ({"weights": [1.0]}, "length"),
        ({"mean": [0.0]}, "length"),
        ({"scale": [1.0]}, "length"),
        ({"scale": [1.0, 0.0]}, "strictly positive"),
        ({"scale": [1.0, float("nan")]}, "NaN/inf"),
        ({"weights": [float("inf"), 0.0]}, "NaN/inf"),
        ({"bias": float("nan")}, "bias"),
        ({"threshold": 1.5}, "threshold"),
        ({"feature_names": ["pair_candidate_count", "pair_candidate_count"]}, "duplicates"),
        ({"feature_names": ["nope", "pair_candidate_count"]}, "not in V232_PAIR_FEATURES"),
        ({"weights": "abc"}, "numeric"),
    ]:
        with pytest.raises(ValueError, match=msg):
            LinearPairGateModel.from_dict(model_dict(**bad))


def test_model_requires_all_keys_and_rejects_unknown():
    data = model_dict()
    del data["threshold"]
    with pytest.raises(ValueError, match="missing keys"):
        LinearPairGateModel.from_dict(data)
    with pytest.raises(ValueError, match="unknown keys"):
        LinearPairGateModel.from_dict(model_dict(extra=1))
    with pytest.raises(ValueError, match="mapping"):
        LinearPairGateModel.from_dict([1, 2])


def test_missing_and_nan_feature_at_inference_fail():
    model = LinearPairGateModel.from_dict(model_dict())
    with pytest.raises(ValueError, match="Missing pair feature"):
        model.logit({"pair_max_candidate_score": 1.0})
    with pytest.raises(ValueError, match="Non-finite"):
        model.logit({"pair_max_candidate_score": float("nan"), "pair_candidate_count": 1.0})


def test_threshold_behavior_and_model_roundtrip():
    cfg = PairGateConfig(enabled=True, model=model_dict())
    gate = PairChangeGate(cfg)
    hi = gate.evaluate({"pair_max_candidate_score": 1.0, "pair_candidate_count": 3.0})
    lo = gate.evaluate({"pair_max_candidate_score": 0.0, "pair_candidate_count": 3.0})
    assert hi.keep_pair and hi.reason is None and hi.raw_score == pytest.approx(1.0)
    assert not lo.keep_pair and lo.reason == "pair_probability_below_threshold"
    # logit exactly 0 -> probability 0.5 >= threshold 0.5 keeps (>= semantics)
    edge = gate.evaluate({"pair_max_candidate_score": 0.5, "pair_candidate_count": 0.0})
    assert edge.probability == pytest.approx(0.5) and edge.keep_pair
    again = LinearPairGateModel.from_dict(gate.model.to_dict())
    assert np.allclose(again.weights, gate.model.weights) and again.feature_names == gate.model.feature_names


def test_enabled_requires_model_and_config_validation():
    with pytest.raises(ValueError, match="requires a model"):
        PairChangeGate(PairGateConfig(enabled=True))
    with pytest.raises(ValueError, match="Unknown pair_gate keys"):
        PairGateConfig.from_mapping({"enabled": False, "bogus": 1})
    with pytest.raises(ValueError, match="top_k"):
        PairGateConfig(top_k=0)
    assert PairGateConfig.from_mapping(None).enabled is False


def test_pair_gate_config_metadata_and_fingerprint():
    """Item 4: PairGateConfig metadata and fingerprint determinism."""
    cfg = PairGateConfig(top_k=5, high_confidence_threshold=0.6, image_size=256)
    meta = cfg.metadata()
    assert meta["schema_version"] == PAIR_GATE_VERSION
    assert meta["top_k"] == 5
    assert meta["high_confidence_threshold"] == 0.6
    assert meta["image_size"] == 256
    assert meta["feature_names"] == list(V232_PAIR_FEATURES)

    fp = cfg.fingerprint()
    assert isinstance(fp, str) and len(fp) == 16
    assert fp == PairGateConfig(top_k=5, high_confidence_threshold=0.6, image_size=256).fingerprint()
    assert fp != PairGateConfig(top_k=3).fingerprint()


def test_model_fingerprint_binding_and_mismatch_rejection():
    """Item 4: serialized model expected fingerprint must match runtime config."""
    cfg = PairGateConfig(top_k=3, high_confidence_threshold=0.5, image_size=256)
    valid_fp = cfg.fingerprint()

    # Model carrying matching fingerprint succeeds
    m_ok = model_dict(expected_fingerprint=valid_fp)
    gate_ok = PairChangeGate(cfg, LinearPairGateModel.from_dict(m_ok))
    assert gate_ok.model.expected_fingerprint == valid_fp

    # Model carrying mismatching fingerprint raises clear ValueError
    m_bad = model_dict(expected_fingerprint="mismatched_hash_999")
    with pytest.raises(ValueError, match="expected fingerprint 'mismatched_hash_999' does not match"):
        PairChangeGate(cfg, LinearPairGateModel.from_dict(m_bad))


def test_pair_training_contract():
    """Item 10: OOF-only pair training contract verification."""
    # Valid contract
    c = PairTrainingContract(
        source_type="oof",
        evidence_classifier_source="v231_ablation_d_cv5",
        notes="5-fold OOF predictions on train set",
    )
    assert c.source_type == "oof"
    assert c.allow_partial_labels is False
    d = c.to_dict()
    assert d["source_type"] == "oof"

    # In-sample source is strictly forbidden
    with pytest.raises(ValueError, match="Invalid pair training source_type: 'in_sample'"):
        PairTrainingContract(source_type="in_sample", evidence_classifier_source="ablation_d")

    # Partial labels as pair negatives is forbidden
    with pytest.raises(ValueError, match="allow_partial_labels=True is prohibited"):
        PairTrainingContract(source_type="oof", evidence_classifier_source="ablation_d", allow_partial_labels=True)

    # Missing evidence classifier source is forbidden
    with pytest.raises(ValueError, match="evidence_classifier_source"):
        PairTrainingContract(source_type="oof", evidence_classifier_source="")


def test_disabled_gate_preserves_output_object():
    gate = PairChangeGate(PairGateConfig())
    feats = extractor().extract({}, {})
    result = gate.evaluate(feats)
    assert result.keep_pair and result.reason == "gate_disabled"
    row = {"id": "a", "new_building": "[1]", "tree_removal": "[2]"}
    assert apply_pair_gate(row, result) is row


def test_reject_clears_target_classes_only():
    gate = PairChangeGate(PairGateConfig(enabled=True, model=model_dict()))
    result = gate.evaluate({"pair_max_candidate_score": 0.0, "pair_candidate_count": 0.0})
    row = {"id": "a", "new_building": "[1]", "tree_removal": "[2]"}
    out = apply_pair_gate(row, result)
    assert out == {"id": "a", "new_building": "", "tree_removal": ""}
    assert row["new_building"] == "[1]"  # input not mutated
