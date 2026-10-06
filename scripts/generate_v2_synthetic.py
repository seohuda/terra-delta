#!/usr/bin/env python3
"""Generate deterministic single-temporal synthetic change pairs from NAIP source tiles."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, optimize=True)


def components(mask, min_area=80, max_area=20000):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            result.append(labels == index)
    return result


def soften(mask, sigma=2.0):
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.clip(alpha[..., None], 0, 1)


def donor_replace(image, mask, rng, donor_ok=None):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    box_h, box_w = y1 - y0, x1 - x0
    if box_h < 2 or box_w < 2 or box_h >= height or box_w >= width:
        return None
    candidates = []
    for _ in range(100):
        dy = int(rng.integers(0, max(1, height - box_h + 1)))
        dx = int(rng.integers(0, max(1, width - box_w + 1)))
        donor_slice = slice(dy, dy + box_h), slice(dx, dx + box_w)
        if donor_ok is not None and donor_ok[donor_slice].mean() < .65:
            continue
        if mask[donor_slice].mean() > .08:
            continue
        candidates.append((dy, dx))
        if len(candidates) >= 8:
            break
    if not candidates:
        return None
    dy, dx = candidates[int(rng.integers(len(candidates)))]
    donor = image[dy:dy + box_h, dx:dx + box_w].copy()
    output = image.copy()
    alpha = soften(mask[y0:y1, x0:x1], sigma=max(1.0, min(box_h, box_w) / 18.0))
    target = output[y0:y1, x0:x1].astype(np.float32)
    mixed = target * (1 - alpha) + donor.astype(np.float32) * alpha
    output[y0:y1, x0:x1] = np.clip(mixed, 0, 255).astype(np.uint8)
    return output


def canopy_and_donor(image, nir):
    red = image[..., 0].astype(np.float32)
    green = image[..., 1].astype(np.float32)
    near_ir = nir.astype(np.float32)
    ndvi = (near_ir - red) / (near_ir + red + 1.0)
    canopy = (ndvi > .20) & (green > red * .85)
    canopy = cv2.morphologyEx(
        canopy.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ) > 0
    donor = (ndvi < .12) & (image.mean(2) > 35)
    return canopy, donor


def organic_mask(canopy, rng):
    ys, xs = np.where(canopy)
    if len(xs) < 400:
        return None
    for _ in range(60):
        k = int(rng.integers(len(xs)))
        cy, cx = int(ys[k]), int(xs[k])
        canvas = np.zeros(canopy.shape, np.uint8)
        for _ in range(int(rng.integers(3, 8))):
            dx, dy = rng.normal(0, 12, 2)
            ax, ay = rng.integers(8, 30, 2)
            center = (
                int(np.clip(cx + dx, 0, canopy.shape[1] - 1)),
                int(np.clip(cy + dy, 0, canopy.shape[0] - 1)),
            )
            cv2.ellipse(
                canvas, center, (int(ax), int(ay)), float(rng.uniform(0, 180)), 0, 360, 255, -1
            )
        mask = (canvas > 0) & canopy
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        ) > 0
        area = int(mask.sum())
        if 180 <= area <= 15000:
            return mask
    return None


def source_rows(root, regions):
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        metadata = folder / "metadata.json"
        if not metadata.is_file():
            continue
        info = json.loads(metadata.read_text())
        region = info.get("region_id") or folder.name.rsplit("_r", 1)[0]
        if region not in regions:
            continue
        post = folder / "post.png"
        nir = folder / "post_nir.png"
        footprint = folder / "building_footprint_reference.png"
        if post.is_file() and nir.is_file() and footprint.is_file():
            rows.append((folder.name, region, post, nir, footprint))
    return rows


def save_sample(output, sid, pre, post, building, tree, region, category, source_id):
    folder = output / sid
    folder.mkdir(parents=True, exist_ok=False)
    write_png(folder / "pre.png", pre)
    write_png(folder / "post.png", post)
    write_png(folder / "new_building.png", building.astype(np.uint8) * 255)
    write_png(folder / "tree_removal.png", tree.astype(np.uint8) * 255)
    return {
        "id": sid,
        "pre": str((folder / "pre.png").resolve()),
        "post": str((folder / "post.png").resolve()),
        "new_building": str((folder / "new_building.png").resolve()),
        "tree_removal": str((folder / "tree_removal.png").resolve()),
        "valid_mask": "",
        "review_mask": "",
        "region_id": region,
        "source": "NAIP-synthetic",
        "category": category,
        "source_id": source_id,
        "license_status": "approved",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--building", type=int, default=700)
    parser.add_argument("--tree", type=int, default=700)
    parser.add_argument("--both", type=int, default=100)
    parser.add_argument("--negative", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    root = Path(args.source_root)
    output = Path(args.output)
    manifest = Path(args.manifest)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    sources = source_rows(root, set(args.regions))
    if len(sources) < 20:
        raise SystemExit(f"not enough sources: {len(sources)}")

    rng = np.random.default_rng(args.seed)
    rows, rejects = [], {}
    targets = [
        ("new_building", args.building),
        ("tree_removal", args.tree),
        ("both", args.both),
    ]
    cache = {}

    def load_source(entry):
        key = entry[0]
        if key not in cache:
            image = rgb(entry[2])
            nir = gray(entry[3])
            footprint = gray(entry[4]) > 0
            canopy, donor = canopy_and_donor(image, nir)
            cache[key] = (image, footprint, canopy, donor)
            if len(cache) > 96:
                cache.pop(next(iter(cache)))
        return cache[key]

    def reject(reason):
        rejects[reason] = rejects.get(reason, 0) + 1

    serial = 0
    for category, count in targets:
        made, attempts = 0, 0
        while made < count and attempts < count * 100:
            attempts += 1
            entry = sources[int(rng.integers(len(sources)))]
            image, footprint, canopy, donor = load_source(entry)
            building = np.zeros(footprint.shape, bool)
            tree = np.zeros(footprint.shape, bool)
            pre, post = image.copy(), image.copy()

            if category in {"new_building", "both"}:
                choices = components(footprint, 100, 16000)
                if not choices:
                    reject("no_building_component")
                    continue
                building = choices[int(rng.integers(len(choices)))]
                low_structure = ~cv2.dilate(
                    footprint.astype(np.uint8), np.ones((7, 7), np.uint8)
                ).astype(bool)
                changed = donor_replace(pre, building, rng, low_structure)
                if changed is None:
                    reject("building_donor")
                    continue
                pre = changed

            if category in {"tree_removal", "both"}:
                safe_canopy = canopy & ~cv2.dilate(
                    footprint.astype(np.uint8), np.ones((9, 9), np.uint8)
                ).astype(bool)
                tree = organic_mask(safe_canopy, rng)
                if tree is None:
                    reject("no_canopy_blob")
                    continue
                changed = donor_replace(post, tree, rng, donor)
                if changed is None:
                    reject("tree_donor")
                    continue
                post = changed

            if building.any() and tree.any():
                tree &= ~building
            serial += 1
            sid = f"syn_{category}_{serial:05d}"
            rows.append(save_sample(
                output, sid, pre, post, building, tree, entry[1], category, entry[0]
            ))
            made += 1

        if made != count:
            raise RuntimeError(f"generated only {made}/{count} {category} samples")

    zero = np.zeros((256, 256), np.uint8)
    for index in range(args.negative):
        entry = sources[index % len(sources)]
        sid = f"neg_{index:05d}_{entry[0]}"
        folder = output / sid
        folder.mkdir(parents=True, exist_ok=False)
        write_png(folder / "new_building.png", zero)
        write_png(folder / "tree_removal.png", zero)
        rows.append({
            "id": sid,
            "pre": str(entry[2].resolve()),
            "post": str(entry[2].resolve()),
            "new_building": str((folder / "new_building.png").resolve()),
            "tree_removal": str((folder / "tree_removal.png").resolve()),
            "valid_mask": "",
            "review_mask": "",
            "region_id": entry[1],
            "source": "NAIP-hard-negative",
            "category": "negative",
            "source_id": entry[0],
            "license_status": "approved",
        })

    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "pre", "post", "new_building", "tree_removal", "valid_mask", "review_mask",
        "region_id", "source", "category", "source_id", "license_status",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "seed": args.seed,
        "regions": args.regions,
        "sources": len(sources),
        "rows": len(rows),
        "counts": {
            key: sum(row["category"] == key for row in rows)
            for key in ("new_building", "tree_removal", "both", "negative")
        },
        "rejects": rejects,
    }
    manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
