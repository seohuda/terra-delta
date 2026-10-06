"""Experimental v2.3.2 feature schemas. The frozen V2.3.1 schemas are never modified.

``V232_CANDIDATE_SCHEMA_D_ALIGN`` is a NEW ordered schema (ablation D followed by alignment
features) for a future candidate classifier; the V2.3.1 ablation D model keeps loading with its
own ``feature_names`` and simply ignores the extra alignment keys.
"""
from __future__ import annotations

from .evidence_features import ABLATION_SCHEMAS
from .local_alignment import V232_ALIGNMENT_FEATURES
from .pair_gate import V232_PAIR_FEATURES

V232_CANDIDATE_SCHEMA_D_ALIGN: tuple[str, ...] = ABLATION_SCHEMAS["D"] + V232_ALIGNMENT_FEATURES

__all__ = ["V232_ALIGNMENT_FEATURES", "V232_PAIR_FEATURES", "V232_CANDIDATE_SCHEMA_D_ALIGN"]
