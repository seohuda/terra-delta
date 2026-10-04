"""Offline Siamese-v2 inference: independent pixels, presence and polygons."""

from collections.abc import Mapping

import numpy as np
import torch

from terradelta.models.siamese_v2 import TerraDeltaSiameseV2, load_v2_checkpoint
from terradelta.postprocess.polygons import mask_to_polygons, serialize_polygons
from terradelta.utils.io import write_prediction_csv
from .predictor import discover_pairs, prepare_pair, read_image

CLASSES = ("new_building", "tree_removal")


def prepare_v2_pair(pre, post):
    """Match the float32 RGB normalization used to train the saved model."""
    return prepare_pair(pre.astype(np.float32), post.astype(np.float32))


def model_options(config):
    options = dict(config.get("model", {}))
    if options.pop("architecture", "siamese_v2") != "siamese_v2":
        raise ValueError("Expected siamese_v2 architecture")
    if options.pop("encoder_weights", None) is not None:
        raise ValueError("V2 inference cannot download encoder weights")
    if set(options) - {"alignment", "alignment_scales", "max_offset"}:
        raise ValueError("Unknown v2 model options")
    return options


def class_options(config, name):
    options = config.get("postprocess", {})
    if options.get("mode", "independent") != "independent":
        raise ValueError("V2 masks must remain independent")
    result = {
        "presence_threshold": 0.5,
        "pixel_threshold": 0.5,
        "min_area": 30,
        "min_pos_area": 20,
        "simplify_px": 0.5,
        "ndigits": 2,
    }
    result.update({key: options[key] for key in result if key in options})
    result.update(options.get("classes", {}).get(name, {}))
    for key in ("presence_threshold", "pixel_threshold"):
        value = result[key]
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name}.{key} must be in [0,1]")
    if result["pixel_threshold"] is None:
        raise ValueError("Pixel threshold cannot be disabled")
    return result


def output_row(identifier, segmentation, presence, config):
    segmentation, presence = np.asarray(segmentation), np.asarray(presence)
    if segmentation.shape != (2, 256, 256) or presence.shape != (2,):
        raise ValueError("Expected 2x256x256 pixel and two presence probabilities")
    for values in (segmentation, presence):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Probabilities must be finite in [0,1]")
    row = {"id": identifier}
    for channel, name in enumerate(CLASSES):
        options = class_options(config, name)
        threshold = options["presence_threshold"]
        if threshold is not None and presence[channel] < threshold:
            row[name] = ""
            continue
        mask = segmentation[channel] >= options["pixel_threshold"]
        polygons = mask_to_polygons(
            mask,
            min_area=options["min_area"],
            min_pos_area=options["min_pos_area"],
            simplify_px=options["simplify_px"],
            ndigits=options["ndigits"],
            backend="reference",
        )
        row[name] = serialize_polygons(polygons)
    return row


class V2Predictor:
    def __init__(self, checkpoint, config, device="cpu", model=None):
        self.config = config
        self.device = torch.device(device)
        for name in CLASSES:
            class_options(config, name)
        self.model = model if model is not None else TerraDeltaSiameseV2(**model_options(config))
        if checkpoint is not None:
            self.metadata = load_v2_checkpoint(checkpoint, self.model)
        elif model is None:
            raise ValueError("Inference needs a checkpoint")
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def probabilities(self, image):
        if image.ndim != 4 or image.shape[1:] != (6, 256, 256) or not torch.isfinite(image).all():
            raise ValueError("Expected finite normalized B6x256x256 paired RGB")
        output = self.model(image.to(self.device))
        if not isinstance(output, Mapping):
            raise ValueError("V2 must return segmentation and presence logits")
        pixels = output["segmentation"].float().sigmoid()
        presence = output["presence"].float().sigmoid()
        if pixels.shape != (len(image), 2, 256, 256) or presence.shape != (len(image), 2):
            raise ValueError("V2 returned incompatible output shapes")
        if not torch.isfinite(pixels).all() or not torch.isfinite(presence).all():
            raise ValueError("V2 returned nonfinite probabilities")
        return pixels.cpu().numpy(), presence.cpu().numpy()


def predict_directory(input_dir, output_path, checkpoint, config, device="cpu"):
    root, ids = discover_pairs(input_dir)
    predictor = V2Predictor(checkpoint, config, device)
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
        rows.extend(output_row(i, p, q, config) for i, p, q in zip(batch_ids, pixels, presence))
        print(f"Predicted {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
    write_prediction_csv(output_path, rows)
    return rows
