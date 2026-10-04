"""Local/offline inference using the same RGB preparation as the baseline."""
from pathlib import Path
import io
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch

from terradelta.models.factory import build_model
from terradelta.models.checkpoint import load_checkpoint
from terradelta.utils.io import read_csv, safe_id, write_prediction_csv
from .alignment import align_pair
from .tta import predict_probabilities

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def read_image(path, size=(256, 256)):
    with Image.open(io.BytesIO(Path(path).read_bytes())) as im:
        arr = np.asarray(im.convert("RGB"))
    if arr.shape[:2] != tuple(size):
        raise ValueError(f"{path}: expected image size {size}, got {arr.shape[:2]}")
    return arr


def prepare_pair(pre, post, alignment="none", alignment_options=None):
    pre, post = align_pair(pre, post, alignment, **(alignment_options or {}))
    image = np.concatenate([(pre / 255.0 - MEAN) / STD, (post / 255.0 - MEAN) / STD], axis=2)
    return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)


class Predictor:
    def __init__(self, model, config=None, device="cpu"):
        self.config = config or {}
        self.options = self.config.get("inference", self.config)
        self.device = device
        self.model = model.to(device).eval()
        self.transforms = self.options.get("tta", ["identity"])
        if self.transforms is False or self.transforms == []:
            self.transforms = ["identity"]
        self.alignment = self.options.get("alignment", "none")

    @torch.inference_mode()
    def predict_batch(self, image):
        if image.ndim != 4 or image.shape[1] != 6:
            raise ValueError("Expected B,6,H,W input")
        # Keep optional registration in one owner so validation and submission
        # process normalized tensors identically. Inversion recovers uint8 RGB.
        if self.alignment != "none":
            pairs = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            rgb_pairs = np.clip(np.rint((pairs * np.tile(STD, 2) + np.tile(MEAN, 2)) * 255), 0, 255).astype(np.uint8)
            image = torch.stack([prepare_pair(pair[..., :3], pair[..., 3:], self.alignment,
                self.options.get("alignment_options")) for pair in rgb_pairs])
        result = predict_probabilities(self.model, image.to(self.device), self.transforms)
        if result.shape != (image.shape[0], 3, *image.shape[2:]) or not torch.isfinite(result).all():
            raise ValueError("Model must return finite B,3,H,W logits at input resolution")
        return result.cpu().numpy()


def discover_pairs(input_dir):
    input_dir = Path(input_dir)
    lists = sorted(input_dir.rglob("pairs.csv"))
    if not lists:
        archives = sorted(input_dir.rglob("*.zip"))
        if len(archives) == 1:
            dst = Path(tempfile.mkdtemp(prefix="terradelta-input-"))
            with zipfile.ZipFile(archives[0]) as archive:
                total = sum(info.file_size for info in archive.infolist())
                if total > 8 * 1024**3:
                    raise ValueError("Input archive exceeds 8 GiB expansion limit")
                for info in archive.infolist():
                    if not (dst / info.filename).resolve().is_relative_to(dst.resolve()):
                        raise ValueError("Unsafe ZIP member")
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("ZIP symlinks are forbidden")
                archive.extractall(dst)
            lists = sorted(dst.rglob("pairs.csv"))
    if len(lists) != 1:
        raise ValueError(f"Expected exactly one pairs.csv; found {len(lists)}")
    rows = read_csv(lists[0])
    ids = [safe_id(row["id"]) for row in rows if row.get("id", "").strip()]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate input ids")
    return lists[0].parent, ids


def predict_directory(input_dir, output_path, checkpoint, config, device="cpu", probability_dir=None):
    from terradelta.postprocess.polygons import predictions_to_polygons, serialize_polygons
    root, ids = discover_pairs(input_dir)
    # Never initialize inference from external pretrained weights.
    model_config = dict(config.get("model", {}), encoder_weights=None)
    model = build_model(model_config)
    load_checkpoint(checkpoint, model)
    predictor = Predictor(model, config, device)
    options = config.get("inference", {})
    batch_size = int(options.get("batch_size", 16))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = []
    if probability_dir is not None:
        probability_dir = Path(probability_dir)
        probability_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        image = torch.stack([prepare_pair(read_image(root / "images" / i / "pre.png"),
            read_image(root / "images" / i / "post.png")) for i in batch_ids])
        probabilities = predictor.predict_batch(image)
        for sample_id, prob in zip(batch_ids, probabilities):
            if probability_dir is not None:
                np.save(probability_dir / f"{sample_id}.npy", prob, allow_pickle=False)
            polygons = predictions_to_polygons(prob, config.get("postprocess", {}))
            rows.append({"id": sample_id, **{k: serialize_polygons(v) for k, v in polygons.items()}})
        print(f"Predicted {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
    write_prediction_csv(output_path, rows)
    return rows
