#!/usr/bin/env python3
"""Package and cleanroom-verify TerraDelta V3 Satlas release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terradelta.data.dataset_v2 import read_v2_manifest
from terradelta.models.satlas_v3 import SatlasV3Model


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_predict_notebook(config_path: Path) -> dict:
    """Construct predict.ipynb for Satlas V3 offline inference."""
    code = r"""# TerraDelta V3 Offline Satlas Inference
import os
import sys
from pathlib import Path

# Locate root directory
def _submission_root():
    anchors = []
    for name in ("__file__", "__vsc_ipynb_file__", "__notebook_path__"):
        value = globals().get(name)
        if value:
            anchors.append(Path(value).resolve().parent)
    for name in ("AIF_SUBMISSION_DIR", "AIF_NOTEBOOK_PATH"):
        value = os.environ.get(name)
        if value:
            path = Path(value).resolve()
            anchors.append(path.parent if path.suffix in {".ipynb", ".py"} else path)
    for value in sys.argv:
        if value.endswith((".ipynb", ".py")) and Path(value).is_file():
            anchors.append(Path(value).resolve().parent)
    anchors.extend((Path("/aif/submission"), Path.cwd()))
    for candidate in anchors:
        if (candidate / "assets/model/model.pt").is_file():
            return candidate
    return Path.cwd()

ROOT = _submission_root()
sys.path.insert(0, str(ROOT / "assets/code"))

import csv
import numpy as np
from PIL import Image
import torch
import yaml
from terradelta.models.satlas_v3 import SatlasV3Model
from terradelta.postprocess.polygons import mask_to_polygons, serialize_polygons

INPUT_DIR = Path(os.environ.get("AIF_INPUT_DIR", "./input"))
PREDICTION_PATH = Path(os.environ.get("AIF_PREDICTION_PATH", "./prediction.csv"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(ROOT / "assets/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Initialize Model
model = SatlasV3Model(weights_path=None, fpn_channels=128, num_classes=2)
ckpt = torch.load(ROOT / "assets/model/model.pt", map_location=DEVICE)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# Locate pairs
pairs = []
pairs_csv = INPUT_DIR / "pairs.csv"
if pairs_csv.is_file():
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r.get("id") or r.get("pair_id")
            if pid:
                pre_p = INPUT_DIR / "images" / pid / "pre.png"
                post_p = INPUT_DIR / "images" / pid / "post.png"
                if not pre_p.is_file():
                    pre_p = INPUT_DIR / pid / "pre.png"
                    post_p = INPUT_DIR / pid / "post.png"
                pairs.append((pid, pre_p, post_p))
else:
    # Scan subdirectories
    for d in sorted(INPUT_DIR.iterdir()):
        if d.is_dir() and (d / "pre.png").is_file() and (d / "post.png").is_file():
            pairs.append((d.name, d / "pre.png", d / "post.png"))

print(f"Loaded {len(pairs)} pairs for inference on {DEVICE}.")

pixel_thresh = CONFIG.get("postprocess", {}).get("classes", {}).get("new_building", {}).get("pixel_threshold", 0.35)
pres_thresh = CONFIG.get("postprocess", {}).get("classes", {}).get("new_building", {}).get("presence_threshold", 0.30)
tree_pixel_thresh = CONFIG.get("postprocess", {}).get("classes", {}).get("tree_removal", {}).get("pixel_threshold", 0.70)
tree_pres_thresh = CONFIG.get("postprocess", {}).get("classes", {}).get("tree_removal", {}).get("presence_threshold", 0.30)

rows = []
batch_size = int(CONFIG.get("inference", {}).get("batch_size", 8))

with torch.inference_mode():
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        tensors = []
        valid_chunk = []
        for pid, pre_path, post_path in chunk:
            try:
                pre = np.array(Image.open(pre_path).convert("RGB"), dtype=np.uint8)
                post = np.array(Image.open(post_path).convert("RGB"), dtype=np.uint8)
                pre_t = torch.from_numpy(pre.transpose(2, 0, 1)).float() / 255.0
                post_t = torch.from_numpy(post.transpose(2, 0, 1)).float() / 255.0
                tensors.append(torch.cat([pre_t, post_t], dim=0))
                valid_chunk.append((pid, pre.shape[:2]))
            except Exception as e:
                print(f"Error loading {pid}: {e}")
                rows.append({"id": pid, "new_building": "EMPTY", "tree_removal": "EMPTY"})

        if not tensors:
            continue

        images = torch.stack(tensors).to(DEVICE)
        seg_logits, pres_logits = model(images)
        seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
        pres_probs = torch.sigmoid(pres_logits).cpu().numpy()

        for idx, (pid, shape) in enumerate(valid_chunk):
            row = {"id": pid}
            # Building
            b_mask = seg_probs[idx, 0] >= pixel_thresh
            if pres_probs[idx, 0] < pres_thresh:
                b_mask[:] = False
            b_polys = mask_to_polygons(b_mask, min_area=30, min_pos_area=20, simplify_px=0.5, backend="reference")
            row["new_building"] = serialize_polygons(b_polys)

            # Tree
            t_mask = seg_probs[idx, 1] >= tree_pixel_thresh
            if pres_probs[idx, 1] < tree_pres_thresh:
                t_mask[:] = False
            t_polys = mask_to_polygons(t_mask, min_area=20, min_pos_area=20, simplify_px=0.5, backend="reference")
            row["tree_removal"] = serialize_polygons(t_polys)

            rows.append(row)

PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(PREDICTION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "new_building", "tree_removal"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Inference completed. Wrote {len(rows)} predictions to {PREDICTION_PATH}.")
"""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            }
        ],
        "metadata": {
            "language_info": {"name": "python", "version": "3.11.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def create_package(
    model_checkpoint: Path,
    config: dict,
    code_dir: Path,
    output_zip: Path,
) -> Path:
    temp_dir = Path("/opt/dlami/nvme/tmp_package")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    assets_dir = temp_dir / "assets"
    assets_code = assets_dir / "code" / "terradelta"
    assets_model = assets_dir / "model"
    assets_model.mkdir(parents=True, exist_ok=True)
    assets_code.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save clean stripped model (removes 720MB optimizer state)
    ckpt = torch.load(model_checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    torch.save({"model_state_dict": state_dict}, assets_model / "model.pt")

    # 2. Write config
    with open(assets_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # 3. Copy code
    shutil.copytree(code_dir, assets_code, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 4. Generate predict.ipynb
    nb = build_predict_notebook(assets_dir / "config.yaml")
    with open(temp_dir / "predict.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # 5. Write requirements.txt
    req_text = "torch>=2.5\ntorchvision>=0.20\nsegmentation-models-pytorch>=0.5\nnumpy>=2.0\npillow>=10.0\nshapely>=2.0\npyyaml>=6\n"
    with open(temp_dir / "requirements.txt", "w") as f:
        f.write(req_text)

    # 6. Make ZIP
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                full_p = Path(root) / file
                rel_p = full_p.relative_to(temp_dir)
                z.write(full_p, rel_p)

    shutil.rmtree(temp_dir)
    return output_zip


def verify_cleanroom(zip_path: Path, test_manifest: Path) -> dict:
    """Extract zip into clean temp room and verify 2-pass byte identical execution."""
    t0 = time.time()
    clean_dir = Path("/opt/dlami/nvme/cleanroom_test")
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(clean_dir)

    # Create dummy input from 5 test pairs
    input_dir = clean_dir / "input"
    input_dir.mkdir(exist_ok=True)
    rows = read_v2_manifest(test_manifest)[:5]

    for r in rows:
        pair_dir = input_dir / "images" / r["id"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r["pre"], pair_dir / "pre.png")
        shutil.copyfile(r["post"], pair_dir / "post.png")

    with open(input_dir / "pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id"])
        writer.writeheader()
        writer.writerows({"id": r["id"]} for r in rows)

    # Pass 1: Run inference
    pass1_pred = clean_dir / "pred1.csv"
    env1 = os.environ.copy()
    env1["AIF_INPUT_DIR"] = str(input_dir)
    env1["AIF_PREDICTION_PATH"] = str(pass1_pred)
    nb_cmd = [
        "/opt/pytorch/bin/python", "-c",
        f"import nbformat, io; from nbconvert.preprocessors import ExecutePreprocessor; "
        f"nb = nbformat.read('{clean_dir}/predict.ipynb', as_version=4); "
        f"ep = ExecutePreprocessor(timeout=600, kernel_name='python3'); "
        f"ep.preprocess(nb, {{'metadata': {{'path': '{clean_dir}'}}}})"
    ]
    subprocess.run(nb_cmd, env=env1, check=True)
    hash1 = compute_sha256(pass1_pred)

    # Pass 2: Repeat inference
    pass2_pred = clean_dir / "pred2.csv"
    env2 = os.environ.copy()
    env2["AIF_INPUT_DIR"] = str(input_dir)
    env2["AIF_PREDICTION_PATH"] = str(pass2_pred)
    subprocess.run(nb_cmd, env=env2, check=True)
    hash2 = compute_sha256(pass2_pred)

    byte_identical = (hash1 == hash2)
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)

    shutil.rmtree(clean_dir)
    return {
        "byte_identical": byte_identical,
        "pass1_sha256": hash1,
        "pass2_sha256": hash2,
        "zip_size_mb": round(zip_size_mb, 2),
        "elapsed_sec": round(time.time() - t0, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Package TerraDelta V3 Satlas")
    parser.add_argument("--model-checkpoint", type=str, default="/opt/dlami/nvme/outputs/v3_satlas/S1/model.pt")
    parser.add_argument("--output-zip", type=str, default="/opt/dlami/nvme/outputs/v3_release/terradelta-v3-satlas.zip")
    parser.add_argument("--test-manifest", type=str, default="/data/terradelta/processed/micro-pilot-v2/val.csv")
    args = parser.parse_args()

    model_ckpt = Path(args.model_checkpoint)
    out_zip = Path(args.output_zip)
    code_dir = Path("/opt/dlami/nvme/terra-delta/src/terradelta")

    config = {
        "architecture": "satlas_v3_swinb",
        "postprocess": {
            "classes": {
                "new_building": {"pixel_threshold": 0.35, "presence_threshold": 0.30},
                "tree_removal": {"pixel_threshold": 0.70, "presence_threshold": 0.30},
            }
        },
        "inference": {"batch_size": 8},
    }

    print(f"Creating release package from {model_ckpt}...")
    create_package(model_ckpt, config, code_dir, out_zip)
    zip_hash = compute_sha256(out_zip)
    zip_size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Created {out_zip}: {zip_size_mb:.2f} MB, SHA256: {zip_hash}")

    print("Running 2-pass cleanroom verification...")
    cleanroom_res = verify_cleanroom(out_zip, Path(args.test_manifest))
    print(f"Cleanroom verification results: {cleanroom_res}")

    if not cleanroom_res["byte_identical"]:
        raise ValueError("Cleanroom verification failed: Predictions are not byte-identical!")

    summary = {
        "status": "PROMOTED_VERIFIED",
        "zip_path": str(out_zip),
        "zip_size_mb": zip_size_mb,
        "zip_sha256": zip_hash,
        "cleanroom": cleanroom_res,
    }
    with open(out_zip.parent / "release_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Release package successfully verified and ready!")


if __name__ == "__main__":
    main()
