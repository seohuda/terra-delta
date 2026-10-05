"""Unit tests for the v2.3.2 pair-level change gate (synthetic data only)."""
import math

import numpy as np
import pytest

from terradelta.inference.pair_gate import (
    CROSS_CLASS_FEATURES,
    PAIR_GATE_CLASSES,
    SCOPE_STEMS,
    SCOPES,
    V232_PAIR_FEATURES,
    LinearPairGateModel,
    PairCandidate,
    PairChangeGate,
    PairFeatureExtractor,
    PairGateConfig,
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


def test_unknown_class_and_nonfinite_candidate_rejected():
    with pytest.raises(ValueError, match="Unknown classes"):
        extractor().extract({"road": [cand()]}, {})
    with pytest.raises(ValueError, match="Non-finite"):
        extractor().extract({"new_building": [cand(score=float("nan"))]}, {})
    with pytest.raises(ValueError, match="Non-finite"):
        extractor().extract({}, {"new_building": float("inf")})


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
