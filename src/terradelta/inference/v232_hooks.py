"""Glue for the experimental v2.3.2 features, kept apart from the frozen V2.3.1 predictor code.

Everything here is pure (no model access) so it can be unit-tested with synthetic arrays.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .alignment_cache import AlignmentCacheKey, AlignmentFeatureCache
from .local_alignment import (
    V232_ALIGNMENT_FEATURES,
    LocalAlignmentConfig,
    LocalAlignmentFeatureExtractor,
    mask_bbox,
    mask_fingerprint,
)
from .pair_gate import (
    PAIR_GATE_CLASSES,
    PairChangeGate,
    PairFeatureExtractor,
    PairGateResult,
    apply_pair_gate,
    candidate_from_record,
)


def validate_classifier_alignment_requirements(
    classifier: Any,
    alignment_config: LocalAlignmentConfig,
) -> None:
    """Ensure that if the active classifier declares any alignment feature, alignment is enabled and fingerprint matches."""
    if classifier is None or not getattr(classifier, "enabled", False):
        return
    models = getattr(classifier, "models", {})
    alignment_feature_set = frozenset(V232_ALIGNMENT_FEATURES)
    active_alignment_models = []
    for class_name, model in models.items():
        declared = set(getattr(model, "feature_names", ()))
        req_align = declared & alignment_feature_set
        if req_align:
            active_alignment_models.append((class_name, sorted(req_align), model))

    if not active_alignment_models:
        return

    if not alignment_config.enabled:
        first_cls, first_feats, _ = active_alignment_models[0]
        raise ValueError(
            f"Classifier model for '{first_cls}' declares experimental alignment features {first_feats}, "
            "but experimental.alignment_residual.enabled is False. Alignment features cannot be silently defaulted to zero."
        )

    expected_fp = getattr(classifier, "expected_alignment_fingerprint", None)
    if not expected_fp:
        for _, _, mod in active_alignment_models:
            mod_fp = getattr(mod, "expected_alignment_fingerprint", None)
            if mod_fp:
                expected_fp = mod_fp
                break

    if not expected_fp or not isinstance(expected_fp, str) or not expected_fp.strip():
        raise ValueError(
            "Classifier uses alignment features but expected_alignment_fingerprint is missing or empty."
        )

    runtime_fp = alignment_config.fingerprint()
    if expected_fp != runtime_fp:
        raise ValueError(
            f"Classifier expected alignment fingerprint '{expected_fp}' does not match "
            f"runtime configuration fingerprint '{runtime_fp}'"
        )


def check_candidate_alignment_features(
    features: Mapping[str, float],
    model: Any,
    class_name: str,
) -> None:
    """Verify that every alignment feature declared by the model is present and finite."""
    if model is None:
        return
    declared = set(getattr(model, "feature_names", ()))
    req_align = declared & frozenset(V232_ALIGNMENT_FEATURES)
    for name in req_align:
        if name not in features:
            raise ValueError(
                f"Candidate for '{class_name}' is missing required alignment feature '{name}'. "
                "Missing alignment features cannot be silently replaced with zero."
            )
        val = features[name]
        if not math.isfinite(val):
            raise ValueError(
                f"Candidate for '{class_name}' has non-finite alignment feature '{name}': {val}"
            )


def attach_alignment_features(
    records: Mapping[str, Sequence[dict[str, Any]]],
    masks: Mapping[str, Sequence[np.ndarray]],
    pre_feats: Mapping[int, np.ndarray],
    post_feats: Mapping[int, np.ndarray],
    extractor: LocalAlignmentFeatureExtractor,
    pair_id: str = "",
    cache: AlignmentFeatureCache | None = None,
    model_fingerprint: str = "",
    pair_fingerprint: str = "",
) -> None:
    """Add alignment features to ``rec["features"]`` in place; existing keys are never altered.

    ``masks[class][k]`` is the boolean image-resolution mask of ``records[class][k]``;
    ``pre_feats``/``post_feats`` map scale index -> ``(C, H, W)`` arrays for ONE pair.
    """
    for name, class_records in records.items():
        class_masks = masks.get(name, [])
        if len(class_masks) != len(class_records):
            raise ValueError(
                f"Candidate mask count ({len(class_masks)}) differs from record count "
                f"({len(class_records)}) for {name}"
            )
        for rec, mask in zip(class_records, class_masks):
            key = None
            bbox = mask_bbox(mask)
            if cache is not None and bbox is not None:
                if not model_fingerprint or not pair_fingerprint:
                    raise ValueError(
                        "Persistent cache reuse requires non-empty model_fingerprint and pair_fingerprint"
                    )
                key = AlignmentCacheKey.build(
                    pair_id,
                    name,
                    rec["component_index"],
                    bbox,
                    mask_fingerprint(mask),
                    extractor.config,
                    model_fingerprint=model_fingerprint,
                    pair_fingerprint=pair_fingerprint,
                )
            values = extractor.extract(mask, pre_feats, post_feats, cache=cache, cache_key=key)
            clash = set(values) & set(rec["features"])
            if clash:
                raise ValueError(f"Alignment features collide with existing features: {sorted(clash)}")
            rec["features"].update(values)


def candidate_score(classifier: Any, class_name: str, features: Mapping[str, float]) -> float:
    """Object evidence probability; falls back to mean segmentation probability w/o a model."""
    model = classifier.models.get(class_name) if getattr(classifier, "enabled", False) else None
    if model is None:
        return float(features.get("mean_probability", 0.0))
    check_candidate_alignment_features(features, model, class_name)
    return float(model.predict_proba(model.extract_vector(features)))


def gate_filtered_row(
    filtered_row: Mapping[str, Any],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    presence: Mapping[str, float],
    classifier: Any,
    pair_features: PairFeatureExtractor,
    gate: PairChangeGate,
) -> tuple[Mapping[str, Any], PairGateResult]:
    """Run the optional pair gate on an already evidence-filtered row.

    A candidate counts as kept only if the classifier keeps it AND the class output of the
    filtered row is still non-empty (min-area rules may have cleared the whole class).
    """
    candidates = {}
    for name in PAIR_GATE_CLASSES:
        class_alive = bool(filtered_row.get(name))
        model = classifier.models.get(name) if getattr(classifier, "enabled", False) else None
        class_candidates = []
        for rec in records.get(name, []):
            feats = rec["features"]
            if model is not None:
                check_candidate_alignment_features(feats, model, name)
            class_candidates.append(
                candidate_from_record(
                    rec,
                    candidate_score(classifier, name, feats),
                    class_alive and classifier.predict_keep(feats, name),
                )
            )
        candidates[name] = class_candidates
    features = pair_features.extract(candidates, presence)
    result = gate.evaluate(features)
    return apply_pair_gate(filtered_row, result), result
