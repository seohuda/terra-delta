#!/usr/bin/env python3
"""Prepare TerraDelta V3 training crops and manifests (S0, S1, S2).

Produces:
1. Event-centered 256x256 positive crops from AIHub 71363 SkySat pairs (new_building).
2. Diverse 256x256 no-change crops from AIHub 71363 SkySat negatives (roofs, roads, infrastructure).
3. Manifests:
   - train_S0.csv: Control (existing clean TerraDelta data only)
   - train_S1.csv: S0 + AIHub positive building crops
   - train_S2.csv: S1 + diverse AIHub negative crops
4. Enforces strict tree-masking rule: all AIHub crops have valid_mask_tree = False, presence_valid_tree = False.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image


def extract_positive_crops(
    pos_pairs: list[dict],
    raw_img_dir: Path,
    raw_lbl_dir: Path,
    output_dir: Path,
    min_change_px: int = 20,
) -> list[dict]:
    """Extract 256x256 crops containing building change from positive pairs."""
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in pos_pairs:
        pair_id = p["pair_id"]
        pre_img_path = raw_img_dir / p["pre_image_file"]
        post_img_path = raw_img_dir / p["post_image_file"]
        pre_lbl_path = raw_lbl_dir / p["pre_mask_file"]
        post_lbl_path = raw_lbl_dir / p["post_mask_file"]

        if not (pre_img_path.exists() and post_img_path.exists() and pre_lbl_path.exists() and post_lbl_path.exists()):
            continue

        im_pre = Image.open(pre_img_path).convert("RGB")
        im_post = Image.open(post_img_path).convert("RGB")
        lbl_pre = np.array(Image.open(pre_lbl_path))
        lbl_post = np.array(Image.open(post_lbl_path))

        b_change = (~(lbl_pre == 10)) & (lbl_post == 10)

        # Stride 256 non-overlapping grid on 1024x1024
        for r in range(0, 1024, 256):
            for c in range(0, 1024, 256):
                patch_change = b_change[r : r + 256, c : c + 256]
                change_px = int(np.sum(patch_change))
                if change_px < min_change_px:
                    continue

                crop_id = f"{pair_id}_r{r}_c{c}"
                crop_folder = output_dir / crop_id
                crop_folder.mkdir(parents=True, exist_ok=True)

                crop_pre = im_pre.crop((c, r, c + 256, r + 256))
                crop_post = im_post.crop((c, r, c + 256, r + 256))
                crop_mask = (patch_change.astype(np.uint8) * 255)
                empty_tree = np.zeros((256, 256), dtype=np.uint8)

                pre_file = crop_folder / "pre.png"
                post_file = crop_folder / "post.png"
                bldg_file = crop_folder / "new_building.png"
                tree_file = crop_folder / "tree_removal.png"

                crop_pre.save(pre_file)
                crop_post.save(post_file)
                Image.fromarray(crop_mask).save(bldg_file)
                Image.fromarray(empty_tree).save(tree_file)

                records.append({
                    "id": crop_id,
                    "pre": str(pre_file.resolve()),
                    "post": str(post_file.resolve()),
                    "new_building": str(bldg_file.resolve()),
                    "tree_removal": str(tree_file.resolve()),
                    "valid_mask": "",
                    "review_mask": "",
                    "region_id": p.get("tile_id", "")[:4],
                    "source": "aihub_71363",
                    "category": "aihub_building",
                    "source_id": p.get("tile_id", ""),
                    "license_status": "aihub_open_korea",
                    "valid_mask_tree": "False",
                    "presence_valid_tree": "False",
                })

    return records


def extract_negative_crops(
    neg_pairs: list[dict],
    raw_img_dir: Path,
    raw_lbl_dir: Path,
    output_dir: Path,
    target_count: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Extract diverse 256x256 no-change crops prioritizing roads/roofs/complex scenes."""
    rng = np.random.default_rng(seed)
    shuffled_pairs = list(neg_pairs)
    rng.shuffle(shuffled_pairs)

    records = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Select scenes with built/infrastructure elements (classes 10, 20, 30, 40)
    for p in shuffled_pairs:
        if len(records) >= target_count:
            break

        pair_id = p["pair_id"]
        pre_img_path = raw_img_dir / p["pre_image_file"]
        post_img_path = raw_img_dir / p["post_image_file"]
        post_lbl_path = raw_lbl_dir / p["post_mask_file"]

        if not (pre_img_path.exists() and post_img_path.exists() and post_lbl_path.exists()):
            continue

        lbl_post = np.array(Image.open(post_lbl_path))
        built_mask = (lbl_post == 10) | (lbl_post == 20) | (lbl_post == 30) | (lbl_post == 40)

        # Check grid positions
        best_pos = None
        max_built = 0
        for r in range(0, 1024, 256):
            for c in range(0, 1024, 256):
                built_count = int(np.sum(built_mask[r : r + 256, c : c + 256]))
                if built_count > max_built and built_count >= 30:
                    max_built = built_count
                    best_pos = (r, c)

        if best_pos is None:
            continue

        r, c = best_pos
        crop_id = f"{pair_id}_neg_r{r}_c{c}"
        crop_folder = output_dir / crop_id
        crop_folder.mkdir(parents=True, exist_ok=True)

        im_pre = Image.open(pre_img_path).convert("RGB")
        im_post = Image.open(post_img_path).convert("RGB")

        crop_pre = im_pre.crop((c, r, c + 256, r + 256))
        crop_post = im_post.crop((c, r, c + 256, r + 256))
        empty_mask = np.zeros((256, 256), dtype=np.uint8)

        pre_file = crop_folder / "pre.png"
        post_file = crop_folder / "post.png"
        bldg_file = crop_folder / "new_building.png"
        tree_file = crop_folder / "tree_removal.png"

        crop_pre.save(pre_file)
        crop_post.save(post_file)
        Image.fromarray(empty_mask).save(bldg_file)
        Image.fromarray(empty_mask).save(tree_file)

        records.append({
            "id": crop_id,
            "pre": str(pre_file.resolve()),
            "post": str(post_file.resolve()),
            "new_building": str(bldg_file.resolve()),
            "tree_removal": str(tree_file.resolve()),
            "valid_mask": "",
            "review_mask": "",
            "region_id": p.get("tile_id", "")[:4],
            "source": "aihub_71363",
            "category": "aihub_negative",
            "source_id": p.get("tile_id", ""),
            "license_status": "aihub_open_korea",
            "valid_mask_tree": "False",
            "presence_valid_tree": "False",
        })

    # Pass 2: If we still need more to reach target_count, take high-variance center crops
    if len(records) < target_count:
        for p in shuffled_pairs:
            if len(records) >= target_count:
                break
            crop_id = f"{p['pair_id']}_neg_center"
            if any(r["id"] == crop_id for r in records):
                continue
            pre_img_path = raw_img_dir / p["pre_image_file"]
            post_img_path = raw_img_dir / p["post_image_file"]
            if not (pre_img_path.exists() and post_img_path.exists()):
                continue

            r, c = 384, 384
            crop_folder = output_dir / crop_id
            crop_folder.mkdir(parents=True, exist_ok=True)

            im_pre = Image.open(pre_img_path).convert("RGB")
            im_post = Image.open(post_img_path).convert("RGB")
            crop_pre = im_pre.crop((c, r, c + 256, r + 256))
            crop_post = im_post.crop((c, r, c + 256, r + 256))
            empty_mask = np.zeros((256, 256), dtype=np.uint8)

            pre_file = crop_folder / "pre.png"
            post_file = crop_folder / "post.png"
            bldg_file = crop_folder / "new_building.png"
            tree_file = crop_folder / "tree_removal.png"

            crop_pre.save(pre_file)
            crop_post.save(post_file)
            Image.fromarray(empty_mask).save(bldg_file)
            Image.fromarray(empty_mask).save(tree_file)

            records.append({
                "id": crop_id,
                "pre": str(pre_file.resolve()),
                "post": str(post_file.resolve()),
                "new_building": str(bldg_file.resolve()),
                "tree_removal": str(tree_file.resolve()),
                "valid_mask": "",
                "review_mask": "",
                "region_id": p.get("tile_id", "")[:4],
                "source": "aihub_71363",
                "category": "aihub_negative",
                "source_id": p.get("tile_id", ""),
                "license_status": "aihub_open_korea",
                "valid_mask_tree": "False",
                "presence_valid_tree": "False",
            })

    return records


def write_csv_manifest(records: list[dict], output_path: Path) -> None:
    """Write records to CSV manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "pre", "post", "new_building", "tree_removal",
        "valid_mask", "review_mask", "region_id", "source",
        "category", "source_id", "license_status",
        "valid_mask_tree", "presence_valid_tree"
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Prepare TerraDelta V3 training crops and manifests")
    parser.add_argument("--aihub-dir", type=str, default="/data/terradelta/incoming/aihub-71363")
    parser.add_argument("--base-manifest", type=str, default="/data/terradelta/v2/manifests/train-v2-realboost.csv")
    parser.add_argument("--crops-dir", type=str, default="/opt/dlami/nvme/data/aihub_crops")
    parser.add_argument("--manifests-dir", type=str, default="/opt/dlami/nvme/data/v3_manifests")
    args = parser.parse_args()

    t0 = time.time()
    aihub_dir = Path(args.aihub_dir)
    raw_img_dir = aihub_dir / "raw" / "VS_02__Skyset"
    raw_lbl_dir = aihub_dir / "raw" / "VL_01_LABEL_02__Skyset"
    scanned_file = aihub_dir / "scanned_pairs.json"

    with scanned_file.open() as f:
        scanned = json.load(f)

    pos_pairs = scanned["pos_building"]
    neg_pairs = scanned["pure_neg"]
    print(f"Loaded {len(pos_pairs)} positive pairs, {len(neg_pairs)} pure negative pairs from {scanned_file}")

    crops_dir = Path(args.crops_dir)
    print(f"Extracting positive crops to {crops_dir}...")
    pos_crops = extract_positive_crops(pos_pairs, raw_img_dir, raw_lbl_dir, crops_dir / "positive", min_change_px=20)
    print(f"Extracted {len(pos_crops)} positive building crops.")

    print(f"Extracting diverse negative crops...")
    neg_crops = extract_negative_crops(neg_pairs, raw_img_dir, raw_lbl_dir, crops_dir / "negative", target_count=100)
    print(f"Extracted {len(neg_crops)} diverse negative crops.")

    # Load base clean manifest
    base_manifest = Path(args.base_manifest)
    with base_manifest.open(newline="", encoding="utf-8-sig") as f:
        base_records = list(csv.DictReader(f))
    print(f"Loaded {len(base_records)} base clean records from {base_manifest}")

    # For base records, ensure valid_mask_tree is True
    for r in base_records:
        r["valid_mask_tree"] = "True"
        r["presence_valid_tree"] = "True"

    # S0: Control (Base only)
    manifests_dir = Path(args.manifests_dir)
    s0_records = list(base_records)
    s0_path = manifests_dir / "train_S0.csv"
    write_csv_manifest(s0_records, s0_path)
    print(f"Wrote S0 manifest: {len(s0_records)} records to {s0_path}")

    # S1: S0 + AIHub Positives
    s1_records = list(base_records) + pos_crops
    s1_path = manifests_dir / "train_S1.csv"
    write_csv_manifest(s1_records, s1_path)
    print(f"Wrote S1 manifest: {len(s1_records)} records to {s1_path}")

    # S2: S1 + Diverse AIHub Negatives
    s2_records = list(base_records) + pos_crops + neg_crops
    s2_path = manifests_dir / "train_S2.csv"
    write_csv_manifest(s2_records, s2_path)
    print(f"Wrote S2 manifest: {len(s2_records)} records to {s2_path}")

    summary = {
        "s0_count": len(s0_records),
        "s1_count": len(s1_records),
        "s2_count": len(s2_records),
        "aihub_pos_crops": len(pos_crops),
        "aihub_neg_crops": len(neg_crops),
        "base_clean_count": len(base_records),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    summary_path = manifests_dir / "v3_manifests_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Manifest preparation completed in {summary['elapsed_sec']}s: {summary}")


if __name__ == "__main__":
    main()
