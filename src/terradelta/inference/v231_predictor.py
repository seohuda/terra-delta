"""TerraDelta v2.3.1 predictor: object-level evidence classifier on v2.2 candidates.

Pipeline:
V2.2 candidate components
-> extract multi-group evidence features (confidence, geometry, TTA, reverse, CVA, RGB)
-> object evidence classifier (pure NumPy)
-> keep/reject
-> original identity polygon output
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from terradelta.postprocess.polygons import reference_components
from terradelta.utils.io import write_prediction_csv
from .evidence_classifier import EvidenceClassifier, apply_evidence_filtering
from .evidence_features import (
    compute_global_metrics,
    compute_reverse_features,
    extract_candidate_evidence_record,
    extract_deep_cva_maps,
    get_component_pixels,
)
from .predictor import discover_pairs, read_image
from .stability_v23 import (
    component_features as stability_component_features,
    tta_probabilities,
)
from .v2 import CLASSES, V2Predictor, class_options, prepare_v2_pair


def reverse_pair_tensor(image: torch.Tensor) -> torch.Tensor:
    """Swap PRE and POST channels for reverse-time inference: (POST, PRE)."""
    # image is (B, 6, 256, 256) where [:3] is PRE and [3:] is POST
    return torch.cat([image[:, 3:], image[:, :3]], dim=1)


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert normalized B6x256x256 tensor back to unnormalized RGB in [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[None, :, None, None]
    pre = (image[:, :3] * std + mean).clamp(0.0, 1.0)
    post = (image[:, 3:] * std + mean).clamp(0.0, 1.0)
    return pre.cpu().numpy(), post.cpu().numpy()


@torch.inference_mode()
def extract_pair_evidence_records(
    predictor: V2Predictor,
    image_batch: torch.Tensor,
    pixels_batch: np.ndarray,
    presence_batch: np.ndarray,
    v22_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    include_reverse: bool = True,
    include_cva: bool = True,
    include_rgb: bool = True,
) -> list[dict[str, list[dict[str, Any]]]]:
    """Extract candidate records with evidence features for each pair in batch."""
    b = len(image_batch)
    # Check if any pair has any candidate
    has_candidates = [any(row.get(c) for c in CLASSES) for row in v22_rows]
    if not any(has_candidates):
        return [{c: [] for c in CLASSES} for _ in range(b)]

    # 1. Multi-view TTA probabilities: shape (4, B, 2, 256, 256) and (4, B, 2)
    tta_pixels, tta_heads = tta_probabilities(predictor, image_batch, (pixels_batch, presence_batch))

    # 2. Reverse inference: 1 forward pass
    if include_reverse:
        rev_images = reverse_pair_tensor(image_batch)
        rev_pixels, rev_heads = predictor.probabilities(rev_images)
    else:
        rev_pixels = rev_heads = None

    # 3. Deep CVA maps from Siamese encoder
    if include_cva:
        cva_maps, cos_maps = extract_deep_cva_maps(
            predictor.model, image_batch[:, :3], image_batch[:, 3:], predictor.device
        )
    else:
        cva_maps = cos_maps = None

    # 4. RGB unnormalized in [0, 1]
    if include_rgb:
        pre_rgbs, post_rgbs = tensor_to_rgb(image_batch)
    else:
        pre_rgbs = post_rgbs = None

    results = []
    for i in range(b):
        pair_records = {}
        if not has_candidates[i]:
            results.append({c: [] for c in CLASSES})
            continue

        global_metrics = compute_global_metrics(pre_rgbs[i], post_rgbs[i]) if include_rgb else None

        # Stability base features for candidate tracking
        stab_features = stability_component_features(
            tta_pixels[:, i], tta_heads[:, i], config
        )

        for channel, name in enumerate(CLASSES):
            options = class_options(config, name)
            candidates = stab_features[name]
            if not candidates or not v22_rows[i].get(name):
                pair_records[name] = []
                continue

            originals = [
                p
                for p in reference_components(pixels_batch[i, channel] >= options["pixel_threshold"])
                if p.area >= options["min_area"]
            ]
            if sum(p.area for p in originals) < options["min_pos_area"]:
                pair_records[name] = []
                continue

            # Reverse candidate components for matching
            if include_reverse and rev_pixels is not None:
                rev_prob = rev_pixels[i, channel]
                rev_pres = float(rev_heads[i, channel])
                rev_parts = [
                    p
                    for p in reference_components(rev_prob >= options["pixel_threshold"])
                    if p.area >= options["min_area"]
                ]
            else:
                rev_prob = np.zeros((256, 256), dtype=np.float32)
                rev_pres = 0.0
                rev_parts = []

            class_records = []
            for k, (part, stab_rec) in enumerate(zip(originals, candidates)):
                y, x = get_component_pixels(part)
                rev_rec = (
                    compute_reverse_features(
                        y, x, part, rev_prob, rev_pres, rev_parts, options["pixel_threshold"]
                    )
                    if include_reverse
                    else None
                )

                rec = extract_candidate_evidence_record(
                    poly=part,
                    component_index=k,
                    y=y,
                    x=x,
                    all_polys=originals,
                    identity_prob=pixels_batch[i, channel],
                    identity_presence=float(presence_batch[i, channel]),
                    verifier_score=0.0,
                    stability_rec=stab_rec,
                    reverse_rec=rev_rec,
                    cva_map=cva_maps[i] if cva_maps is not None else None,
                    cosine_map=cos_maps[i] if cos_maps is not None else None,
                    pre_rgb=pre_rgbs[i] if pre_rgbs is not None else None,
                    post_rgb=post_rgbs[i] if post_rgbs is not None else None,
                    global_metrics=global_metrics,
                    class_name=name,
                )
                class_records.append(rec)
            pair_records[name] = class_records
        results.append(pair_records)

    return results


class V231Predictor(V2Predictor):
    """V2.3.1 predictor: V2.2 pipeline with object-level evidence classifier."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: Mapping[str, Any],
        device: str = "cpu",
        model: torch.nn.Module | None = None,
        classifier: EvidenceClassifier | None = None,
    ):
        super().__init__(checkpoint, config, device, model)
        self.model.requires_grad_(False)
        self.evidence_config = config.get("evidence_classifier", {})
        if classifier is not None:
            self.classifier = classifier
        elif "model_path" in self.evidence_config:
            self.classifier = EvidenceClassifier.from_file(self.evidence_config["model_path"])
        elif "model_data" in self.evidence_config:
            self.classifier = EvidenceClassifier.from_dict(self.evidence_config["model_data"])
        else:
            self.classifier = EvidenceClassifier({}, enabled=False)

    @torch.inference_mode()
    def output_rows(
        self,
        identifiers: Sequence[str],
        image: torch.Tensor,
        pixels: np.ndarray,
        presence: np.ndarray,
    ) -> list[dict[str, Any]]:
        # 1. Base v2.2 predictions
        v22_rows = super().output_rows(identifiers, image, pixels, presence)

        if not self.classifier.enabled or not any(row[c] for row in v22_rows for c in CLASSES):
            return v22_rows

        # 2. Extract evidence features
        evidence_records = extract_pair_evidence_records(
            self, image, pixels, presence, v22_rows, self.config
        )

        # 3. Apply object evidence classifier
        return [
            apply_evidence_filtering(row, recs, self.classifier, self.config)
            for row, recs in zip(v22_rows, evidence_records)
        ]


def predict_directory(input_dir, output_path, checkpoint, config, device="cpu"):
    root, ids = discover_pairs(input_dir)
    predictor = V231Predictor(checkpoint, config, device)
    batch_size = config.get("inference", {}).get("batch_size", 8)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    rows = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        image = torch.stack(
            [
                prepare_v2_pair(
                    read_image(root / "images" / i / "pre.png"), read_image(root / "images" / i / "post.png")
                )
                for i in batch_ids
            ]
        )
        pixels, presence = predictor.probabilities(image)
        rows.extend(predictor.output_rows(batch_ids, image, pixels, presence))
        print(f"Predicted {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
    write_prediction_csv(output_path, rows)
    return rows
