"""Glue for the experimental v2.3.2 features, kept apart from the frozen V2.3.1 predictor code.

Everything here is pure (no model access) so it can be unit-tested with synthetic arrays.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .alignment_cache import AlignmentCacheKey, AlignmentFeatureCache
from .local_alignment import LocalAlignmentFeatureExtractor, mask_bbox, mask_fingerprint
from .pair_gate import (
    PAIR_GATE_CLASSES,
    PairChangeGate,
    PairFeatureExtractor,
    PairGateResult,
    apply_pair_gate,
    candidate_from_record,
)


def attach_alignment_features(
    records: Mapping[str, Sequence[dict[str, Any]]],
    masks: Mapping[str, Sequence[np.ndarray]],
    pre_feats: Mapping[int, np.ndarray],
    post_feats: Mapping[int, np.ndarray],
    extractor: LocalAlignmentFeatureExtractor,
    pair_id: str = "",
    cache: AlignmentFeatureCache | None = None,
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
                key = AlignmentCacheKey.build(
                    pair_id, name, rec["component_index"], bbox, mask_fingerprint(mask), extractor.config
                )
            values = extractor.extract(mask, pre_feats, post_feats, cache=cache, cache_key=key)
            clash = set(values) & set(rec["features"])
            if clash:
                raise ValueError(f"Alignment features collide with existing features: {sorted(clash)}")
            rec["features"].update(values)


def candidate_score(classifier: Any, class_name: str, features: Mapping[str, float]) -> float:
    """Object evidence probability; falls back to mean segmentation probability w/o a model."""
    model = classifier.models.get(class_name) if classifier.enabled else None
    if model is None:
        return float(features.get("mean_probability", 0.0))
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
        candidates[name] = [
            candidate_from_record(
                rec,
                candidate_score(classifier, name, rec["features"]),
                class_alive and classifier.predict_keep(rec["features"], name),
            )
            for rec in records.get(name, [])
        ]
    features = pair_features.extract(candidates, presence)
    result = gate.evaluate(features)
    return apply_pair_gate(filtered_row, result), result
