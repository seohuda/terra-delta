"""Pair-level change gate (v2.3.2 experimental, disabled by default).

Question answered: "does this PRE/POST pair contain convincing evidence that at least one
real target change exists?" The gate sees aggregated candidate evidence for the pair and may
veto the whole pair as an optional final step after the V2.3.1 object evidence classifier.

Only a serialized logistic model is supported (stdlib + NumPy). Nothing here is trained and no
threshold has a default: a model config must supply its own threshold explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

# Must equal terradelta.inference.v2.CLASSES (asserted in tests); duplicated to stay torch-free.
PAIR_GATE_CLASSES: tuple[str, ...] = ("new_building", "tree_removal")
CLASS_PREFIX = {"new_building": "building", "tree_removal": "tree"}

# Per-scope feature stems; scopes are "pair" (all classes pooled), "building" and "tree".
SCOPE_STEMS: tuple[str, ...] = (
    "candidate_count",
    "kept_candidate_count",
    "max_candidate_score",
    "second_candidate_score",
    "mean_top_k_score",
    "score_margin_top1_top2",
    "total_candidate_area",
    "max_candidate_area",
    "mean_candidate_area",
    "total_kept_area",
    "max_tta_stability",
    "mean_tta_stability",
    "max_reverse_support",
    "mean_reverse_support",
    "presence_logit",
    "presence_probability",
    "fraction_of_image_covered",
    "num_high_confidence_candidates",
)
CROSS_CLASS_FEATURES: tuple[str, ...] = (
    "total_candidates_all_classes",
    "max_score_any_class",
    "total_area_all_classes",
    "both_classes_present",
    "building_vs_tree_score_margin",
)
SCOPES: tuple[str, ...] = ("pair", "building", "tree")

V232_PAIR_FEATURES: tuple[str, ...] = (
    tuple(f"{scope}_{stem}" for scope in SCOPES for stem in SCOPE_STEMS) + CROSS_CLASS_FEATURES
)
_KNOWN = frozenset(V232_PAIR_FEATURES)
_PROB_EPS = 1e-6


@dataclass(frozen=True)
class PairCandidate:
    """Minimal per-candidate evidence needed for pair aggregation."""

    score: float  # object evidence probability (or fallback confidence)
    kept: bool  # survived the V2.3.1 object evidence classifier
    area: float  # pixels
    tta_stability: float  # TTA persistence fraction
    reverse_support: float  # reverse-time same-class mean probability
    confidence: float  # forward mean segmentation probability


def candidate_from_record(record: Mapping[str, Any], score: float, kept: bool) -> PairCandidate:
    """Build a :class:`PairCandidate` from a V2.3.1 evidence record's feature dict."""
    f = record["features"]
    return PairCandidate(
        score=float(score),
        kept=bool(kept),
        area=float(f["area"]),
        tta_stability=float(f["persistence_fraction"]),
        reverse_support=float(f["reverse_mean_prob"]),
        confidence=float(f["mean_probability"]),
    )


@dataclass(frozen=True)
class PairGateConfig:
    """``experimental.pair_gate`` config. ``model`` is the serialized logistic model mapping."""

    enabled: bool = False
    top_k: int = 3  # k for mean_top_k_score
    high_confidence_threshold: float = 0.5  # feature definition only, not a decision threshold
    image_size: int = 256
    model: Mapping[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("pair_gate.enabled must be a boolean")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("pair_gate.top_k must be a positive integer")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int) or self.image_size < 1:
            raise ValueError("pair_gate.image_size must be a positive integer")
        thr = self.high_confidence_threshold
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not 0.0 <= thr <= 1.0:
            raise ValueError("pair_gate.high_confidence_threshold must be in [0, 1]")
        if self.model is not None and not isinstance(self.model, Mapping):
            raise ValueError("pair_gate.model must be a mapping")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> PairGateConfig:
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("pair_gate config must be a mapping")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown pair_gate keys: {sorted(unknown)}")
        return cls(**dict(data))


class PairFeatureExtractor:
    """Aggregate candidate evidence into the fixed :data:`V232_PAIR_FEATURES` schema."""

    def __init__(self, config: PairGateConfig):
        self.config = config

    def _scope(
        self, candidates: Sequence[PairCandidate], presence: float
    ) -> dict[str, float]:
        cfg = self.config
        scores = sorted((c.score for c in candidates), reverse=True)
        kept = [c for c in candidates if c.kept]
        areas = [c.area for c in candidates]
        stab = [c.tta_stability for c in candidates]
        rev = [c.reverse_support for c in candidates]
        p = min(max(presence, _PROB_EPS), 1.0 - _PROB_EPS)
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
        total_area = float(sum(areas))
        return {
            "candidate_count": float(len(candidates)),
            "kept_candidate_count": float(len(kept)),
            "max_candidate_score": top1,
            "second_candidate_score": top2,
            "mean_top_k_score": float(np.mean(scores[: cfg.top_k])) if scores else 0.0,
            "score_margin_top1_top2": top1 - top2,
            "total_candidate_area": total_area,
            "max_candidate_area": float(max(areas)) if areas else 0.0,
            "mean_candidate_area": float(np.mean(areas)) if areas else 0.0,
            "total_kept_area": float(sum(c.area for c in kept)),
            "max_tta_stability": float(max(stab)) if stab else 0.0,
            "mean_tta_stability": float(np.mean(stab)) if stab else 0.0,
            "max_reverse_support": float(max(rev)) if rev else 0.0,
            "mean_reverse_support": float(np.mean(rev)) if rev else 0.0,
            "presence_logit": math.log(p / (1.0 - p)),
            "presence_probability": p,
            "fraction_of_image_covered": total_area / float(cfg.image_size**2),
            "num_high_confidence_candidates": float(
                sum(c.confidence >= cfg.high_confidence_threshold for c in candidates)
            ),
        }

    def extract(
        self,
        candidates: Mapping[str, Sequence[PairCandidate]],
        presence: Mapping[str, float],
    ) -> dict[str, float]:
        """Return features in schema order. Missing classes are treated as empty."""
        unknown = set(candidates) - set(PAIR_GATE_CLASSES)
        if unknown:
            raise ValueError(f"Unknown classes in pair candidates: {sorted(unknown)}")
        per_class = {name: list(candidates.get(name, ())) for name in PAIR_GATE_CLASSES}
        pres = {name: float(presence.get(name, 0.0)) for name in PAIR_GATE_CLASSES}
        for cand_list in per_class.values():
            for c in cand_list:
                vals = (c.score, c.area, c.tta_stability, c.reverse_support, c.confidence)
                if not all(math.isfinite(v) for v in vals):
                    raise ValueError("Non-finite candidate evidence")
        if not all(math.isfinite(v) for v in pres.values()):
            raise ValueError("Non-finite presence probability")

        pooled = [c for name in PAIR_GATE_CLASSES for c in per_class[name]]
        scopes = {
            "pair": self._scope(pooled, max(pres.values())),
            "building": self._scope(per_class["new_building"], pres["new_building"]),
            "tree": self._scope(per_class["tree_removal"], pres["tree_removal"]),
        }
        out: dict[str, float] = {}
        for scope in SCOPES:
            for stem in SCOPE_STEMS:
                out[f"{scope}_{stem}"] = scopes[scope][stem]
        b, t = scopes["building"], scopes["tree"]
        out["total_candidates_all_classes"] = float(len(pooled))
        out["max_score_any_class"] = max(b["max_candidate_score"], t["max_candidate_score"])
        out["total_area_all_classes"] = b["total_candidate_area"] + t["total_candidate_area"]
        out["both_classes_present"] = float(bool(per_class["new_building"]) and bool(per_class["tree_removal"]))
        out["building_vs_tree_score_margin"] = b["max_candidate_score"] - t["max_candidate_score"]
        if tuple(out) != V232_PAIR_FEATURES:  # schema/extractor drift guard
            raise RuntimeError("Pair feature order diverged from V232_PAIR_FEATURES")
        return out


def _finite_vector(name: str, values: Any, n: int | None = None) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pair gate model '{name}' must be a numeric list") from exc
    if arr.ndim != 1:
        raise ValueError(f"pair gate model '{name}' must be 1-D")
    if n is not None and len(arr) != n:
        raise ValueError(f"pair gate model '{name}' length {len(arr)} != feature count {n}")
    if not np.isfinite(arr).all():
        raise ValueError(f"pair gate model '{name}' contains NaN/inf")
    return arr


class LinearPairGateModel:
    """Standardized logistic model: sigmoid(((x - mean) / scale) . weights + bias)."""

    REQUIRED = ("feature_names", "mean", "scale", "weights", "bias", "threshold")

    def __init__(
        self,
        feature_names: Sequence[str],
        mean: Sequence[float],
        scale: Sequence[float],
        weights: Sequence[float],
        bias: float,
        threshold: float,
    ):
        names = tuple(feature_names)
        if not names:
            raise ValueError("pair gate model needs at least one feature")
        if not all(isinstance(n, str) for n in names):
            raise ValueError("pair gate feature_names must be strings")
        if len(set(names)) != len(names):
            raise ValueError("pair gate feature_names contain duplicates")
        unknown = [n for n in names if n not in _KNOWN]
        if unknown:
            raise ValueError(f"pair gate feature_names not in V232_PAIR_FEATURES: {unknown}")
        self.feature_names = names
        self.mean = _finite_vector("mean", mean, len(names))
        self.scale = _finite_vector("scale", scale, len(names))
        self.weights = _finite_vector("weights", weights, len(names))
        if (self.scale <= 0).any():
            raise ValueError("pair gate scale values must be strictly positive")
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
            raise ValueError("pair gate bias must be a finite number")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
            raise ValueError("pair gate threshold must be a number in [0, 1]")
        self.bias = float(bias)
        self.threshold = float(threshold)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinearPairGateModel:
        if not isinstance(data, Mapping):
            raise ValueError("pair gate model must be a mapping")
        missing = [k for k in cls.REQUIRED if k not in data]
        if missing:
            raise ValueError(f"pair gate model missing keys: {missing}")
        unknown = set(data) - set(cls.REQUIRED)
        if unknown:
            raise ValueError(f"pair gate model has unknown keys: {sorted(unknown)}")
        return cls(**{k: data[k] for k in cls.REQUIRED})

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
        }

    def vector(self, features: Mapping[str, float]) -> np.ndarray:
        """Feature vector in model order. Missing or non-finite features raise."""
        vec = np.empty(len(self.feature_names), dtype=np.float64)
        for i, name in enumerate(self.feature_names):
            if name not in features:
                raise ValueError(f"Missing pair feature: {name}")
            value = float(features[name])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite pair feature: {name}")
            vec[i] = value
        return vec

    def logit(self, features: Mapping[str, float]) -> float:
        z = (self.vector(features) - self.mean) / self.scale
        return float(np.dot(z, self.weights) + self.bias)

    @staticmethod
    def probability(logit: float) -> float:
        logit = max(-50.0, min(50.0, logit))
        return 1.0 / (1.0 + math.exp(-logit))


@dataclass(frozen=True)
class PairGateResult:
    keep_pair: bool
    probability: float
    raw_score: float
    features: dict
    reason: str | None


class PairChangeGate:
    """Optional final pair veto. When disabled it never alters output."""

    def __init__(self, config: PairGateConfig, model: LinearPairGateModel | None = None):
        self.config = config
        if model is None and config.model is not None:
            model = LinearPairGateModel.from_dict(config.model)
        if config.enabled and model is None:
            raise ValueError("pair_gate.enabled requires a model")
        self.model = model

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def evaluate(self, features: Mapping[str, float]) -> PairGateResult:
        if not self.enabled:
            return PairGateResult(True, 1.0, 0.0, dict(features), "gate_disabled")
        raw = self.model.logit(features)
        prob = self.model.probability(raw)
        keep = prob >= self.model.threshold
        reason = None if keep else "pair_probability_below_threshold"
        return PairGateResult(keep, prob, raw, dict(features), reason)


def apply_pair_gate(row: Mapping[str, Any], result: PairGateResult) -> Mapping[str, Any]:
    """Return ``row`` unchanged on keep; otherwise a copy with every target class emptied."""
    if result.keep_pair:
        return row
    cleared = dict(row)
    for name in PAIR_GATE_CLASSES:
        if name in cleared:
            cleared[name] = ""
    return cleared
