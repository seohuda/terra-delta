"""Independent two-label dataset and target-domain augmentations for TerraDelta v2."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ("new_building", "tree_removal")


def _rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _mask(path, shape):
    with Image.open(path) as image:
        value = np.asarray(image.convert("L")) > 0
    if value.shape != shape:
        raise ValueError(f"mask shape mismatch: {path}")
    return value


def _tensor(image):
    x = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def read_v2_manifest(path):
    path = Path(path).resolve()
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError("empty manifest")
    ids = [row.get("id", "").strip() for row in rows]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("manifest IDs must be nonempty and unique")
    for row in rows:
        for key in ("pre", "post", *CLASSES, "valid_mask", "review_mask"):
            value = (row.get(key) or "").strip()
            if value and value.lower() != "absent":
                p = Path(value)
                row[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    return rows


def _warp(image, matrix, interpolation, border=cv2.BORDER_REFLECT_101):
    h, w = image.shape[:2]
    return cv2.warpAffine(image, matrix, (w, h), flags=interpolation, borderMode=border)


def _photo(image, rng, strength=1.0):
    out = image.astype(np.float32)
    out = (out - 127.5) * rng.uniform(1 - .18 * strength, 1 + .18 * strength) + 127.5
    out += rng.uniform(-28, 28) * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    gamma = rng.uniform(max(.65, 1 - .25 * strength), 1 + .3 * strength)
    lut = np.clip((np.arange(256) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
    out = cv2.LUT(out, lut)
    if rng.random() < .65:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8) * strength) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(.75, 1.25), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < .25:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(.2, 1.0))
    if rng.random() < .25:
        quality = int(rng.integers(60, 94))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if rng.random() < .3:
        haze = rng.uniform(.01, .12) * strength
        out = np.clip(out.astype(np.float32) * (1 - haze) + 235 * haze, 0, 255).astype(np.uint8)
    return out


class V2Transform:
    """Common geometry, then PRE-only relative misregistration and independent radiometry."""

    def __init__(self, *, relative_probability=.55, max_translation=6.0,
                 max_rotation=1.5, max_scale=.02, photometric=True):
        self.relative_probability = float(relative_probability)
        self.max_translation = float(max_translation)
        self.max_rotation = float(max_rotation)
        self.max_scale = float(max_scale)
        self.photometric = bool(photometric)

    def __call__(self, pre, post, masks, valid, seed):
        rng = np.random.default_rng(int(seed) % 2**32)
        pre, post = pre.copy(), post.copy()
        masks, valid = masks.copy(), valid.copy()
        if rng.random() < .5:
            pre, post, masks, valid = np.flip(pre, 1), np.flip(post, 1), np.flip(masks, 2), np.flip(valid, 1)
        if rng.random() < .5:
            pre, post, masks, valid = np.flip(pre, 0), np.flip(post, 0), np.flip(masks, 1), np.flip(valid, 0)
        turns = int(rng.integers(0, 4))
        if turns:
            pre, post = np.rot90(pre, turns), np.rot90(post, turns)
            masks, valid = np.rot90(masks, turns, axes=(1, 2)), np.rot90(valid, turns)
        if rng.random() < self.relative_probability:
            h, w = pre.shape[:2]
            matrix = cv2.getRotationMatrix2D(
                ((w - 1) / 2, (h - 1) / 2),
                rng.uniform(-self.max_rotation, self.max_rotation),
                rng.uniform(1 - self.max_scale, 1 + self.max_scale),
            )
            matrix[:, 2] += rng.uniform(-self.max_translation, self.max_translation, 2)
            pre = _warp(pre, matrix, cv2.INTER_LINEAR)
        if self.photometric:
            pre = _photo(pre, rng)
            post = _photo(post, rng)
        return (np.ascontiguousarray(pre), np.ascontiguousarray(post),
                np.ascontiguousarray(masks), np.ascontiguousarray(valid))


class IndependentChangeDataset(Dataset):
    def __init__(self, manifest, transform=None):
        self.rows = read_v2_manifest(manifest) if not isinstance(manifest, list) else [dict(r) for r in manifest]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def get(self, index, seed=None):
        row = self.rows[index]
        pre, post = _rgb(row["pre"]), _rgb(row["post"])
        if pre.shape != post.shape:
            raise ValueError(f"image mismatch: {row['id']}")
        shape = pre.shape[:2]
        labels = []
        for name in CLASSES:
            value = (row.get(name) or "").strip()
            absent = value.lower() == "absent"
            if absent:
                labels.append(np.zeros(shape, dtype=bool))
            elif value and Path(value).is_file():
                labels.append(_mask(value, shape))
            else:
                raise ValueError(f"{row['id']}: missing {name} label")
        labels = np.stack(labels)
        valid = np.ones(shape, dtype=bool)
        partial = False
        for key in ("valid_mask", "review_mask"):
            value = (row.get(key) or "").strip()
            if value:
                valid &= _mask(value, shape)
                if key == "review_mask":
                    partial = True
        presence = labels.reshape(2, -1).any(1).astype(np.float32)
        presence_valid = np.ones(2, dtype=bool)
        if partial:
            for c, name in enumerate(CLASSES):
                if not presence[c] and (row.get(name) or "").strip().lower() != "absent":
                    presence_valid[c] = False
        if self.transform is not None:
            pre, post, labels, valid = self.transform(pre, post, labels, valid, seed if seed is not None else index)
        pre_t, post_t = _tensor(pre), _tensor(post)
        return {
            "id": row["id"],
            "pre": pre_t,
            "post": post_t,
            "image": torch.cat([pre_t, post_t], 0),
            "target": torch.from_numpy(labels.astype(np.float32)),
            "valid_mask": torch.from_numpy(valid.astype(bool)),
            "presence": torch.from_numpy(presence),
            "presence_valid": torch.from_numpy(presence_valid),
            "category": row.get("category") or row.get("sample_type") or "unknown",
            "region_id": row.get("region_id", ""),
        }

    def __getitem__(self, index):
        return self.get(index, seed=index)
