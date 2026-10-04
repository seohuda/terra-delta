"""The baseline's RGB+RGB input and mutually exclusive change labels.

Transforms consume/return numpy HWC uint8 ``pre``, ``post`` and HW ``mask``.
Normalization is performed here, once, after augmentation. Mask overlap is an
annotation error, not an implicit class-priority rule. License approval belongs
to manifest construction (``pairing.filter_training_rows``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from terradelta.utils.io import safe_id

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MASK_CLASSES = ("new_building", "tree_removal")
MANIFEST_FIELDS = (
    "id", "pre", "post", "new_building", "tree_removal", "region_id", "state",
    "source", "year_pre", "year_post", "license_status", "valid_mask", "review_mask",
)


def read_manifest(path: str | Path) -> list[dict]:
    """Resolve image/mask paths relative to the CSV, never the current directory."""
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"id", "pre", "post"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    try:
        ids = [safe_id(row["id"]) for row in rows]
    except ValueError as error:
        raise ValueError(f"Manifest ids must be nonempty and unique safe sample ids: {error}") from error
    for row, identifier in zip(rows, ids):
        row["id"] = identifier
    if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest ids must be nonempty and unique")
    for row in rows:
        for key in ("pre", "post", *MASK_CLASSES, "valid_mask", "review_mask"):
            value = (row.get(key) or "").strip()
            if value and value.lower() != "absent":
                row[key] = str((path.parent / value).resolve())
    return rows


def _rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _binary(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"1", "L", "I", "I;16", "P"}:
            raise ValueError(f"Expected a single-channel binary mask: {path}")
        mask = np.array(image)
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} differs from image shape {shape}: {path}")
    values = set(np.unique(mask).tolist())
    if not values <= {0, 1, 255, False, True}:
        raise ValueError(f"Mask must contain only 0/1/255: {path}")
    return mask > 0


def _tensor(image: np.ndarray) -> torch.Tensor:
    normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


class ChangeDataset(Dataset):
    """Read a CSV, a sample folder, or a root of sorted sample folders.

    Missing training masks raise. Explicit class absence may be represented by
    the literal ``absent`` in a mask column or ``<class>_absent=true``. Folder
    datasets must contain both masks. ``require_masks=False`` supports inference
    and returns background where a mask is unspecified; ``has_labels`` records
    whether both classes were actually annotated or explicitly absent.
    """

    def __init__(self, manifest, transform=None, require_masks=True, return_auxiliary=False):
        self.transform = transform
        self.require_masks = require_masks
        self.return_auxiliary = return_auxiliary
        root = Path(manifest)
        if root.is_file():
            self.samples = read_manifest(root)
        elif root.is_dir():
            folders = [root] if (root / "pre.png").exists() else sorted(p for p in root.iterdir() if p.is_dir())
            self.samples = [
                {"id": p.name, **{key: str(p / f"{key}.png") for key in ("pre", "post", *MASK_CLASSES)}}
                for p in folders
            ]
        else:
            raise FileNotFoundError(root)
        if not self.samples:
            raise ValueError("Dataset contains no samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        pre, post = _rgb(row["pre"]), _rgb(row["post"])
        if pre.shape != post.shape:
            raise ValueError(f"Pair dimensions differ for {row['id']}: {pre.shape} / {post.shape}")
        shape = pre.shape[:2]
        binaries, annotated = [], []
        for key in MASK_CLASSES:
            value = row.get(key, "")
            absent = str(value).lower() == "absent" or str(row.get(f"{key}_absent", "")).lower() in {"true", "1", "yes"}
            exists = bool(value) and not absent and Path(value).is_file()
            if exists:
                binaries.append(_binary(value, shape))
            elif absent or (not self.require_masks and not value):
                binaries.append(np.zeros(shape, dtype=bool))
            elif not self.require_masks and value and not Path(value).exists():
                # Folder inference contains generated mask paths, but no files.
                binaries.append(np.zeros(shape, dtype=bool))
            else:
                raise ValueError(f"Missing {key} mask for {row['id']}; mark verified absence explicitly")
            annotated.append(exists or absent)
        if np.any(binaries[0] & binaries[1]):
            raise ValueError(f"Overlapping change classes for {row['id']}")
        mask = binaries[0].astype(np.uint8) + 2 * binaries[1].astype(np.uint8)
        validity = np.ones(shape, dtype=bool)
        for key in ("valid_mask", "review_mask"):
            if row.get(key):
                validity &= _binary(row[key], shape)
        if not validity.all():
            mask = mask.astype(np.int16)
            mask[~validity] = -100

        if self.transform is not None:
            result = self.transform(pre=pre, post=post, mask=mask)
            pre, post, mask = result["pre"], result["post"], result["mask"]
        if pre.shape != post.shape or pre.ndim != 3 or pre.shape[2] != 3 or mask.shape != pre.shape[:2]:
            raise ValueError("Transform must return matched HWC RGB pairs and HW labels")
        if not set(np.unique(mask).tolist()) <= {0, 1, 2, -100}:
            raise ValueError("Transform corrupted discrete mask labels")
        pre_tensor, post_tensor = _tensor(pre), _tensor(post)
        output = {
            "id": row["id"], "pre": pre_tensor, "post": post_tensor,
            "image": torch.cat((pre_tensor, post_tensor), dim=0),
            "mask": torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64))),
            "has_labels": all(annotated),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(mask != -100)),
        }
        if self.return_auxiliary:
            output["auxiliary_mask"] = torch.from_numpy(np.stack((mask == 1, mask == 2)))
        return output
