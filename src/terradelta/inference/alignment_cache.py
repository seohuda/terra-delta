"""Cache scaffolding for alignment features (mechanism only; nothing populates it in bulk).

Keys carry everything that defines a value so stale entries cannot be reused silently:
pair/class/component identity, bbox, candidate mask fingerprint, feature scales,
alignment config fingerprint, model/checkpoint fingerprint, and pair/input fingerprint.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .local_alignment import V232_ALIGNMENT_FEATURES, LocalAlignmentConfig

CACHE_VERSION = 2


@dataclass(frozen=True)
class AlignmentCacheKey:
    pair_id: str
    class_name: str
    component_id: int
    bbox: tuple[int, int, int, int]
    candidate_fingerprint: str
    feature_scales: tuple[int, ...]
    config_fingerprint: str
    model_fingerprint: str
    pair_fingerprint: str

    @classmethod
    def build(
        cls,
        pair_id: str,
        class_name: str,
        component_id: int,
        bbox: tuple[int, int, int, int],
        candidate_fingerprint: str,
        config: LocalAlignmentConfig,
        model_fingerprint: str,
        pair_fingerprint: str,
    ) -> AlignmentCacheKey:
        if not model_fingerprint or not isinstance(model_fingerprint, str):
            raise ValueError("AlignmentCacheKey requires a non-empty string model_fingerprint")
        if not pair_fingerprint or not isinstance(pair_fingerprint, str):
            raise ValueError("AlignmentCacheKey requires a non-empty string pair_fingerprint")
        return cls(
            pair_id=str(pair_id),
            class_name=class_name,
            component_id=int(component_id),
            bbox=tuple(int(v) for v in bbox),
            candidate_fingerprint=candidate_fingerprint,
            feature_scales=tuple(config.feature_scales),
            config_fingerprint=config.fingerprint(),
            model_fingerprint=str(model_fingerprint),
            pair_fingerprint=str(pair_fingerprint),
        )

    def as_string(self) -> str:
        return json.dumps(
            [
                self.pair_id,
                self.class_name,
                self.component_id,
                list(self.bbox),
                self.candidate_fingerprint,
                list(self.feature_scales),
                self.config_fingerprint,
                self.model_fingerprint,
                self.pair_fingerprint,
            ]
        )


class AlignmentFeatureCache:
    """In-memory store of aggregated alignment feature dicts with JSON persistence."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, float]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: AlignmentCacheKey) -> dict[str, float] | None:
        value = self._entries.get(key.as_string())
        return dict(value) if value is not None else None

    def put(self, key: AlignmentCacheKey, features: Mapping[str, float]) -> None:
        missing = [n for n in V232_ALIGNMENT_FEATURES if n not in features]
        if missing:
            raise ValueError(f"Cannot cache incomplete alignment features, missing: {missing}")
        cleaned: dict[str, float] = {}
        for n in V232_ALIGNMENT_FEATURES:
            val = float(features[n])
            if not math.isfinite(val):
                raise ValueError(f"Cannot cache non-finite alignment feature '{n}': {val}")
            cleaned[n] = val
        self._entries[key.as_string()] = cleaned

    def save(self, path: str | Path) -> None:
        payload = {"version": CACHE_VERSION, "entries": self._entries}
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AlignmentFeatureCache:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION or not isinstance(data.get("entries"), dict):
            raise ValueError(f"Malformed or outdated alignment cache file (expected version {CACHE_VERSION})")
        cache = cls()
        for key, values in data["entries"].items():
            if set(values) != set(V232_ALIGNMENT_FEATURES):
                raise ValueError("Alignment cache entry has unexpected feature names")
            entry: dict[str, float] = {}
            for n in V232_ALIGNMENT_FEATURES:
                val = float(values[n])
                if not math.isfinite(val):
                    raise ValueError(f"Cached feature '{n}' is non-finite: {val}")
                entry[n] = val
            cache._entries[key] = entry
        return cache
