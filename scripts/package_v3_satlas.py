#!/usr/bin/env python3
"""Build a private Satlas submission and verify it with an explicit local manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from terradelta.utils.io import load_config, read_csv, safe_id  # noqa: E402

MAX_EXPANDED_BYTES = 8 * 1024**3
MAX_ZIP_MEMBERS = 100_000


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_predict_notebook(config_path: Path) -> dict:
    """Keep the historical package anchors; settings live in assets/config.yaml."""
    code = r"""# TerraDelta V3 Offline Satlas Inference
import os
import sys
from pathlib import Path

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

import torch
from terradelta.utils.io import load_config
from terradelta.inference.satlas_v3 import predict_directory

CONFIG = load_config(ROOT / "assets/config.yaml")
INPUT_DIR = Path(os.environ.get("AIF_INPUT_DIR", "./input"))
PREDICTION_PATH = Path(os.environ.get("AIF_PREDICTION_PATH", "./prediction.csv"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rows = predict_directory(INPUT_DIR, PREDICTION_PATH,
    ROOT / "assets/model/model.pt", CONFIG, device=DEVICE)
print("Saved", len(rows), "pairs to", PREDICTION_PATH)
"""
    return {
        "cells": [{"cell_type": "code", "id": "satlas-predict", "execution_count": None,
                   "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
        "metadata": {"language_info": {"name": "python"},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def create_package(model_checkpoint: Path, config: dict, code_dir: Path, output_zip: Path) -> Path:
    """Strip checkpoint metadata without changing tensors; publish without overwrite."""
    output_zip, code_dir = Path(output_zip), Path(code_dir)
    if output_zip.exists() or output_zip.is_symlink():
        raise FileExistsError(output_zip)
    if not (code_dir / "__init__.py").is_file():
        raise ValueError("code_dir must be the terradelta package directory")
    if any(p.is_symlink() for p in (code_dir, *code_dir.rglob("*"))):
        raise ValueError("Package source symlinks are forbidden")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if not (REPO_ROOT / name).is_file():
            raise FileNotFoundError(REPO_ROOT / name)
    notices = REPO_ROOT / "third_party"
    if not notices.is_dir():
        raise FileNotFoundError(notices)
    if any(p.is_symlink() for p in (notices, *notices.rglob("*"))):
        raise ValueError("Notice symlinks are forbidden")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terradelta-package-", dir=output_zip.parent) as temporary:
        staging = Path(temporary)
        root = staging / "package"
        assets = root / "assets"
        (assets / "model").mkdir(parents=True)
        checkpoint = torch.load(model_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, dict) or not state or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError("Checkpoint must contain a nonempty tensor state dictionary")
        torch.save({"model_state_dict": state}, assets / "model/model.pt")
        (assets / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        # Bundle Python sources only, never local datasets, weights or archives.
        for source in sorted(code_dir.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            target = assets / "code/terradelta" / source.relative_to(code_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            shutil.copyfile(REPO_ROOT / name, root / name)
        shutil.copytree(notices, root / "third_party")
        (root / "predict.ipynb").write_text(
            json.dumps(build_predict_notebook(assets / "config.yaml"), indent=2), encoding="utf-8")
        (root / "requirements.txt").write_text(
            "torch>=2.5\ntorchvision>=0.20\nsegmentation-models-pytorch==0.5.0\n"
            "numpy>=2.0\npillow>=10.0\nshapely>=2.0\npyyaml>=6\n",
            encoding="utf-8")
        archive_path = staging / "submission.zip"
        with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        # An exclusive hard link publishes completed bytes and closes the existence-check race.
        os.link(archive_path, output_zip)
    return output_zip


def extract_package(archive: zipfile.ZipFile, destination: Path) -> None:
    """Validate all members before extraction into a newly allocated cleanroom."""
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS or sum(i.file_size for i in members) > MAX_EXPANDED_BYTES:
        raise ValueError("ZIP expansion limit exceeded")
    seen = set()
    for info in members:
        path = PurePosixPath(info.filename)
        if ("\\" in info.filename or path.is_absolute() or ".." in path.parts
                or not path.parts or ":" in path.parts[0]
                or not (destination / info.filename).resolve().is_relative_to(destination.resolve())):
            raise ValueError("Unsafe ZIP member")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("ZIP symlinks and special files are forbidden")
        if path in seen:
            raise ValueError("Duplicate ZIP member")
        seen.add(path)
    archive.extractall(destination)


def verify_cleanroom(zip_path: Path, test_manifest: Path) -> dict:
    """Execute the archived notebook twice in fresh kernels using explicit local inputs."""
    start = time.monotonic()
    test_manifest = Path(test_manifest).resolve()
    rows = read_csv(test_manifest)
    ids = [safe_id(row.get("id", "")) for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Manifest must have nonempty unique IDs")
    rows = rows[:5]
    with tempfile.TemporaryDirectory(prefix="terradelta-cleanroom-") as temporary:
        clean_dir = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            extract_package(archive, clean_dir)
        input_dir = clean_dir / "input"
        for row in rows:
            pair_dir = input_dir / "images" / safe_id(row["id"])
            pair_dir.mkdir(parents=True)
            for key in ("pre", "post"):
                source = Path(row[key])
                if not source.is_absolute():
                    source = test_manifest.parent / source
                shutil.copyfile(source, pair_dir / f"{key}.png")
        with (input_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id"])
            writer.writeheader()
            writer.writerows({"id": row["id"]} for row in rows)
        # Explicit kernel argv ensures use of this environment rather than a global kernelspec.
        runner = """import os, sys, nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
km = KernelManager()
km._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                            display_name='Python', language='python')
nb = nbformat.read('predict.ipynb', as_version=4)
NotebookClient(nb, km=km, timeout=600, resources={'metadata': {'path': os.getcwd()}}).execute()
"""
        hashes = []
        for index in (1, 2):
            prediction = clean_dir / f"pred{index}.csv"
            environment = dict(os.environ, AIF_SUBMISSION_DIR=str(clean_dir),
                               AIF_INPUT_DIR=str(input_dir), AIF_PREDICTION_PATH=str(prediction))
            environment.pop("PYTHONPATH", None)
            subprocess.run([sys.executable, "-c", runner], cwd=clean_dir,
                           env=environment, check=True, timeout=700)
            with prediction.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["id", "new_building", "tree_removal"]:
                    raise ValueError("Invalid prediction CSV schema")
                predictions = list(reader)
            if [r["id"] for r in predictions] != [r["id"] for r in rows]:
                raise ValueError("Prediction IDs do not match the manifest")
            hashes.append(compute_sha256(prediction))
    if hashes[0] != hashes[1]:
        raise ValueError("Cleanroom predictions are not byte-identical")
    return {"byte_identical": True, "pass1_sha256": hashes[0], "pass2_sha256": hashes[1],
            "zip_sha256": compute_sha256(zip_path),
            "zip_size_mb": round(Path(zip_path).stat().st_size / (1024 * 1024), 2),
            "elapsed_sec": round(time.monotonic() - start, 2)}


def main():
    parser = argparse.ArgumentParser(description="Package a private TerraDelta V3 Satlas submission")
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/inference_final.yaml")
    parser.add_argument("--code-dir", type=Path, default=REPO_ROOT / "src/terradelta")
    parser.add_argument("--output-zip", type=Path, default=REPO_ROOT / "outputs/private-submission.zip")
    parser.add_argument("--test-manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.test_manifest.is_file():
        parser.error("--test-manifest must be an existing explicit local manifest")
    package = create_package(args.model_checkpoint, load_config(args.config), args.code_dir, args.output_zip)
    result = verify_cleanroom(package, args.test_manifest)
    print(json.dumps({"zip_path": str(package), "cleanroom": result}, indent=2))


if __name__ == "__main__":
    main()
