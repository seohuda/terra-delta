"""Pure NumPy object-level evidence classifier.

Zero external ML runtime dependencies (no sklearn/LightGBM required at inference).
Evaluates candidate components and makes keep/reject decisions.
Preserves original identity polygon coordinates exactly without deformation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from terradelta.postprocess.polygons import serialize_polygons
from .v2 import CLASSES, class_options


class LinearComponentClassifier:
    """Standardized Logistic Regression evaluator implemented in pure NumPy."""

    def __init__(
        self,
        feature_names: Sequence[str],
        mean: Sequence[float],
        scale: Sequence[float],
        weights: Sequence[float],
        intercept: float,
        threshold: float = 0.5,
    ):
        self.feature_names = tuple(feature_names)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.intercept = float(intercept)
        self.threshold = float(threshold)

        if len(self.mean) != len(self.feature_names) or len(self.scale) != len(self.feature_names):
            raise ValueError("Mean and scale shapes must match feature_names length")
        if len(self.weights) != len(self.feature_names):
            raise ValueError("Weights shape must match feature_names length")
        if (self.scale <= 0).any():
            raise ValueError("Scale values must be strictly positive")

    def extract_vector(self, features: Mapping[str, float]) -> np.ndarray:
        """Extract features in the exact declared order."""
        vec = np.empty(len(self.feature_names), dtype=np.float64)
        for i, name in enumerate(self.feature_names):
            val = features.get(name, 0.0)
            if not np.isfinite(val):
                val = 0.0
            vec[i] = val
        return vec

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """Compute sigmoid probability from standardized feature vector."""
        z = (feature_vector - self.mean) / self.scale
        logit = float(np.dot(z, self.weights) + self.intercept)
        logit = max(-50.0, min(50.0, logit))
        return 1.0 / (1.0 + math_exp(-logit))

    def predict_keep(self, features: Mapping[str, float]) -> bool:
        """Return True to KEEP candidate component, False to REJECT."""
        vec = self.extract_vector(features)
        proba = self.predict_proba(vec)
        return proba >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "logistic_regression",
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinearComponentClassifier:
        return cls(
            feature_names=data["feature_names"],
            mean=data["mean"],
            scale=data["scale"],
            weights=data["weights"],
            intercept=data["intercept"],
            threshold=data.get("threshold", 0.5),
        )


def math_exp(x: float) -> float:
    """Numerically safe exponent."""
    if x > 50.0:
        return np.exp(50.0)
    elif x < -50.0:
        return np.exp(-50.0)
    return float(np.exp(x))


class EvidenceClassifier:
    """Multi-class object evidence classifier container."""

    def __init__(self, models: Mapping[str, LinearComponentClassifier], enabled: bool = True):
        self.models = dict(models)
        self.enabled = bool(enabled)
        for name in self.models:
            if name not in CLASSES:
                raise ValueError(f"Unknown class name: {name}")

    def predict_keep(self, features: Mapping[str, float], class_name: str) -> bool:
        if not self.enabled or class_name not in self.models:
            return True
        return self.models[class_name].predict_keep(features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "enabled": self.enabled,
            "classes": {k: v.to_dict() for k, v in self.models.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceClassifier:
        models = {k: LinearComponentClassifier.from_dict(v) for k, v in data.get("classes", {}).items()}
        return cls(models=models, enabled=data.get("enabled", True))

    @classmethod
    def from_file(cls, path: str | Path) -> EvidenceClassifier:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _polygon_area(poly: Any, rec: Mapping[str, Any] | None = None) -> float:
    """Robustly compute component area from features or geometry."""
    if rec and "features" in rec and "area" in rec["features"]:
        return float(rec["features"]["area"])
    if isinstance(poly, dict):
        from shapely.geometry import shape as shapely_shape
        return float(shapely_shape(poly).area)
    if isinstance(poly, (list, tuple)) and len(poly) >= 3:
        from shapely.geometry import Polygon
        return float(Polygon(poly).area)
    return 0.0


def apply_evidence_filtering(
    row: Mapping[str, Any],
    candidate_records: Mapping[str, Sequence[Mapping[str, Any]]],
    classifier: EvidenceClassifier | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Filter candidate components using evidence classifier.

    Original identity polygons and coordinates are preserved strictly.
    Rejection removes only selected components; it never dilates, shifts or deforms shapes.
    """
    result = dict(row)
    if classifier is None or not classifier.enabled:
        return result

    for name in CLASSES:
        if not row.get(name):
            continue  # Pair already rejected by v2.2 presence/verifier

        records = candidate_records.get(name, [])
        options = class_options(config, name)
        original_polys = json.loads(row[name])

        if len(original_polys) != len(records):
            raise ValueError(
                f"Candidate record count ({len(records)}) differs from identity polygon count ({len(original_polys)})"
            )

        kept_polys = []
        kept_areas = []
        for rec in records:
            features = rec.get("features", {})
            if classifier.predict_keep(features, name):
                kept_polys.append(rec["polygon"])
                kept_areas.append(_polygon_area(rec["polygon"], rec))

        # Check total remaining area constraint
        total_kept_area = sum(kept_areas) if kept_polys else 0.0
        if total_kept_area < options["min_pos_area"] or len(kept_polys) == 0:
            result[name] = ""
        elif len(kept_polys) != len(original_polys):
            result[name] = serialize_polygons(kept_polys)

    return result
