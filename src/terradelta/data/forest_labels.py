"""Conservative high-resolution forest-loss proposals with mandatory review.

Hansen is ONLY a region/candidate mining source, NEVER segmentation truth.
The candidate mask must already be rasterized onto the registered high-res
RGB pair's grid. This module does not resize a 30m loss raster into GT.
RGB greenness is a weak vegetation heuristic: crops, seasons and illumination
can fool it, and it does not identify tree canopy. ``proposed_mask`` and
``confidence`` are review aids, never sufficient training labels.

API: ``ForestLossRefiner().refine(pre, post, candidate_mask,
candidate_source='hansen', valid_mask=None, pre_vegetation=None,
post_vegetation=None, manual_mask=None, review_decision=None, reviewer=None)``.
RGB inputs: HWC uint8; masks: HW uint8/bool; optional vegetation evidence:
HW float probabilities in [0,1] from a HIGH-RESOLUTION canopy/vegetation model.
Returns pre/post, uint8 new_building/tree_removal, proposed_mask, confidence,
valid_mask, review_mask, train_eligible and JSON-compatible metadata.

Default: tree_removal is empty, train_eligible=False, review_status='pending'.
Approval requires ``review_decision='approve'``, a nonempty reviewer identity,
and an explicit ``manual_mask`` (empty masks are allowed). Manual corrections
may differ from the spectral proposal but must stay inside candidate/valid
pixels. Rejection requires ``review_decision='reject'`` and a reviewer; it
creates a reviewed empty tree label. Only these reviewed results are training
eligible, and ONLY within review_mask (candidate & valid), never the whole tile.
Review provenance must be retained by any downstream dataset builder.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .synthetic_building import _image, _mask


def _vegetation(image: np.ndarray) -> np.ndarray:
    # Brightness-normalized green excess; dark pixels are separately excluded.
    rgb = image.astype(np.float32)
    total = np.maximum(rgb.sum(axis=2), 1)
    excess = (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) / total
    return np.clip(excess * 2, 0, 1).astype(np.float32)


def _evidence(value: np.ndarray | None, image: np.ndarray, name: str) -> np.ndarray:
    if value is None:
        return _vegetation(image)
    if (not isinstance(value, np.ndarray) or value.shape != image.shape[:2]
            or not np.issubdtype(value.dtype, np.floating)):
        raise ValueError(f"{name} must be an HW floating probability array on the high-resolution grid")
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} values must be finite probabilities in [0, 1]")
    return value.astype(np.float32, copy=True)


@dataclass(frozen=True)
class ForestLossRefiner:
    """Proposal thresholds are tunable heuristics, not calibrated probabilities.

    Component size and erosion operate in high-resolution pixels. Manual masks
    bypass heuristic filtering because an explicit reviewer is authoritative.
    This operation is deterministic and uses no random state.
    """

    pre_threshold: float = 0.55
    post_threshold: float = 0.25
    min_drop: float = 0.35
    min_component_area: int = 16
    erosion_radius: int = 1
    min_brightness: float = 30.0

    def __post_init__(self) -> None:
        for name in ("pre_threshold", "post_threshold", "min_drop"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.post_threshold >= self.pre_threshold:
            raise ValueError("post_threshold must be below pre_threshold")
        for name, lower in (("min_component_area", 1), ("erosion_radius", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if not np.isfinite(self.min_brightness) or not 0 <= self.min_brightness <= 255:
            raise ValueError("min_brightness must be finite and in [0, 255]")

    def refine(
        self, pre: np.ndarray, post: np.ndarray, candidate_mask: np.ndarray, *,
        candidate_source: str = "hansen", valid_mask: np.ndarray | None = None,
        pre_vegetation: np.ndarray | None = None, post_vegetation: np.ndarray | None = None,
        manual_mask: np.ndarray | None = None, review_decision: str | None = None,
        reviewer: str | None = None,
    ) -> dict:
        pre, post = _image(pre, "pre"), _image(post, "post")
        if pre.shape != post.shape:
            raise ValueError("pre and post must be registered and have identical shapes")
        if not isinstance(candidate_source, str) or not candidate_source.strip():
            raise ValueError("candidate_source must be a nonempty provenance string")
        shape = pre.shape[:2]
        candidate = _mask(candidate_mask, shape, "candidate_mask")
        valid = np.ones(shape, dtype=bool) if valid_mask is None else _mask(valid_mask, shape, "valid_mask")
        review_mask = candidate & valid
        before = _evidence(pre_vegetation, pre, "pre_vegetation")
        after = _evidence(post_vegetation, post, "post_vegetation")
        illuminated = ((pre.mean(axis=2) >= self.min_brightness)
                       & (post.mean(axis=2) >= self.min_brightness))
        drop = before - after
        proposed = (review_mask & illuminated & (before >= self.pre_threshold)
                    & (after <= self.post_threshold) & (drop >= self.min_drop))
        if self.erosion_radius:
            radius = int(self.erosion_radius)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            proposed = cv2.erode(proposed.astype(np.uint8), kernel,
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        count, components, stats, _ = cv2.connectedComponentsWithStats(proposed.astype(np.uint8), 8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_component_area
        proposed = keep[components]
        confidence = np.where(review_mask & illuminated,
                              np.clip(drop, 0, 1) * before * (1 - after), 0).astype(np.float32)
        label = np.zeros(shape, dtype=np.uint8)
        if review_decision is None:
            if manual_mask is not None or reviewer is not None:
                raise ValueError("manual_mask/reviewer require an explicit approve or reject decision")
            review_status = "pending"
            eligible = False
        else:
            if review_decision not in ("approve", "reject"):
                raise ValueError("review_decision must be approve or reject")
            if not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError("manual review requires a nonempty reviewer identity")
            if review_decision == "approve":
                if manual_mask is None:
                    raise ValueError("approval requires an explicit manually reviewed mask")
                reviewed = _mask(manual_mask, shape, "manual_mask")
                if np.any(reviewed & ~review_mask):
                    raise ValueError("manual_mask must be confined to candidate and valid pixels")
                label = reviewed.astype(np.uint8)
                review_status = "approved"
            else:
                if manual_mask is not None and _mask(manual_mask, shape, "manual_mask").any():
                    raise ValueError("rejection cannot carry a positive manual_mask")
                review_status = "rejected"
            eligible = bool(review_mask.any())
        metadata = {
            "source": "forest_loss_refinement", "candidate_source": candidate_source,
            "candidate_only": True, "hansen_direct_gt": False,
            "proposal_method": "high_resolution_vegetation_drop",
            "evidence_source": "rgb_heuristic" if pre_vegetation is None and post_vegetation is None
            else "high_resolution_evidence_and_or_rgb",
            "review_status": review_status, "reviewer": reviewer,
            "label_source": "manual_review" if review_status != "pending" else "unlabeled",
            "train_eligible": eligible, "training_scope": "review_mask_only",
            "candidate_pixels": int(candidate.sum()), "proposal_pixels": int(proposed.sum()),
            "review_pixels": int(review_mask.sum()), "label_pixels": int(label.sum()),
            "confidence_is_calibrated": False,
        }
        return {
            "pre": pre.copy(), "post": post.copy(),
            "new_building": np.zeros(shape, dtype=np.uint8), "tree_removal": label,
            "proposed_mask": proposed.astype(np.uint8), "confidence": confidence,
            "valid_mask": valid.astype(np.uint8), "review_mask": review_mask.astype(np.uint8),
            "train_eligible": eligible, "metadata": metadata,
        }

    __call__ = refine


def refine_forest_loss(pre: np.ndarray, post: np.ndarray, candidate_mask: np.ndarray, **options) -> dict:
    """Functional wrapper using default thresholds; options are refine keyword arguments."""
    return ForestLossRefiner().refine(pre, post, candidate_mask, **options)
