"""Cache scaffolding for alignment features (mechanism only; nothing populates it in bulk).

Keys carry everything that defines a value so stale entries cannot be reused silently:
pair/class/component identity, bbox, candidate mask fingerprint, feature scales and the
alignment config fingerprint (which embeds the algorithm version).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .local_alignment import V232_ALIGNMENT_FEATURES, LocalAlignmentConfig


@dataclass(frozen=True)
class AlignmentCacheKey:
    pair_id: str
    class_name: str
    component_id: int
    bbox: tuple[int, int, int, int]
    candidate_fingerprint: str
    feature_scales: tuple[int, ...]
    config_fingerprint: str

    @classmethod
    def build(
        cls,
        pair_id: str,
        class_name: str,
        component_id: int,
        bbox: tuple[int, int, int, int],
        candidate_fingerprint: str,
        config: LocalAlignmentConfig,
    ) -> AlignmentCacheKey:
        return cls(
            pair_id=str(pair_id),
            class_name=class_name,
            component_id=int(component_id),
            bbox=tuple(int(v) for v in bbox),
            candidate_fingerprint=candidate_fingerprint,
            feature_scales=tuple(config.feature_scales),
            config_fingerprint=config.fingerprint(),
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
        self._entries[key.as_string()] = {n: float(features[n]) for n in V232_ALIGNMENT_FEATURES}

    def save(self, path: str | Path) -> None:
        payload = {"version": 1, "entries": self._entries}
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AlignmentFeatureCache:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("entries"), dict):
            raise ValueError("Malformed alignment cache file")
        cache = cls()
        for key, values in data["entries"].items():
            if set(values) != set(V232_ALIGNMENT_FEATURES):
                raise ValueError("Alignment cache entry has unexpected feature names")
            cache._entries[key] = {n: float(values[n]) for n in V232_ALIGNMENT_FEATURES}
        return cache
