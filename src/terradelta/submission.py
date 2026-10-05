"""Self-contained offline inference exports; no upload or submission API."""
from pathlib import Path
import json
import shutil
import tempfile
import zipfile

import yaml

from terradelta.models.checkpoint import load_checkpoint
from terradelta.models.factory import build_model
from terradelta.utils.io import load_config

PACKAGE_ROOT = Path(__file__).parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def make_notebook(architecture="baseline", *, stability=False):
    if architecture not in {"baseline", "siamese_v2"}:
        raise ValueError("Unknown submission architecture")
    if stability and architecture != "siamese_v2":
        raise ValueError("Component stability requires frozen Siamese v2")
    code = '''# Offline inference. All assets are bundled in this ZIP.
import os
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "assets/config.yaml").is_file():
    raise FileNotFoundError("Run predict.ipynb from the extracted submission root")
sys.path.insert(0, str(ROOT / "assets/code"))
import torch
from terradelta.utils.io import load_config
from terradelta.inference.predictor import predict_directory

CONFIG = load_config(ROOT / "assets/config.yaml")
INPUT_DIR = Path(os.environ.get("AIF_INPUT_DIR", "./input"))
PREDICTION_PATH = Path(os.environ.get("AIF_PREDICTION_PATH", "./prediction.csv"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
'''
    infer = '''# CPU fallback is limited to accelerator readiness, before processing input.
from terradelta.models.factory import build_model
from terradelta.models.checkpoint import load_checkpoint
if DEVICE == "cuda":
    try:
        probe = build_model({"encoder_weights": None}).to(DEVICE).eval()
        load_checkpoint(ROOT / "assets/model/model.pt", probe)
        with torch.inference_mode():
            probe(torch.zeros(1, 6, 256, 256, device=DEVICE))
        del probe
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        print("CUDA readiness failed; using CPU:", type(error).__name__)
        DEVICE = "cpu"
    torch.cuda.empty_cache()
rows = predict_directory(INPUT_DIR, PREDICTION_PATH,
    ROOT / "assets/model/model.pt", CONFIG, device=DEVICE)
print("Saved", len(rows), "pairs to", PREDICTION_PATH)
'''
    if architecture == "siamese_v2":
        code = code.replace("from terradelta.inference.predictor import predict_directory",
                            "from terradelta.inference.v2 import predict_directory")
        code += "\ntorch.set_num_threads(2)\ntorch.use_deterministic_algorithms(True)\ntorch.backends.cudnn.benchmark = False\ntorch.backends.cudnn.deterministic = True\n"
        infer = infer.replace("from terradelta.models.factory import build_model\nfrom terradelta.models.checkpoint import load_checkpoint",
                              "from terradelta.models.siamese_v2 import TerraDeltaSiameseV2, load_v2_checkpoint\nfrom terradelta.inference.v2 import model_options")
        infer = infer.replace('build_model({"encoder_weights": None})', 'TerraDeltaSiameseV2(**model_options(CONFIG))')
        infer = infer.replace("load_checkpoint(", "load_v2_checkpoint(")
        if stability:
            code = code.replace("from terradelta.inference.v2 import predict_directory",
                                "from terradelta.inference.stability_v23 import predict_directory")
    def cell(kind, source):
        c = {"cell_type": kind, "id": "terradelta-" + str(len(source)) + "-" + kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        return c
    return {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [cell("markdown", "# TerraDelta offline prediction\n\nReads AIF_INPUT_DIR and writes AIF_PREDICTION_PATH. No training, network calls or submit commands.\n"),
                  cell("code", code), cell("code", infer)]}


def export_submission(checkpoint, config_path, destination, baseline_root=None):
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination)
    config = load_config(config_path)
    if config.get("model", {}).get("encoder_weights") is not None:
        raise ValueError("Submission must set encoder_weights: null")
    # Validate checkpoint against exact architecture before copying bytes.
    architecture = config.get("model", {}).get("architecture", "baseline")
    if architecture == "siamese_v2":
        from terradelta.inference.v2 import V2Predictor
        predictor_type = V2Predictor
        if "stability" in config:
            from terradelta.inference.stability_v23 import V23Predictor
            predictor_type = V23Predictor
        predictor = predictor_type(checkpoint, config)
        validated_model, metadata = predictor.model, predictor.metadata
        if "optimizer" in metadata or "config" in metadata or "rng" in metadata:
            raise ValueError("V2 deployment checkpoint must contain weights and provenance only")
    elif architecture == "baseline":
        if "stability" in config:
            raise ValueError("Component stability requires frozen Siamese v2")
        validated_model = build_model()
        metadata = load_checkpoint(checkpoint, validated_model)
    else:
        raise ValueError("Unknown submission architecture")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    baseline_root = Path(baseline_root or REPO_ROOT / "baseline/original/03_illegal structure submission")
    for name in ("LICENSE", "NOTICE"):
        if not (baseline_root / name).is_file():
            raise FileNotFoundError(f"Required source notice missing: {baseline_root / name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terradelta-export-", dir=destination.parent) as td:
        root = Path(td) / "package"
        (root / "assets/model").mkdir(parents=True)
        if architecture == "baseline" and ("optimizer" in metadata or "training_format_version" in metadata):
            # Deployment carries weights/provenance only, not optimizer state,
            # RNG, local dataset paths or the full training configuration.
            import torch
            from terradelta import CLASSES
            payload = {"state_dict": validated_model.state_dict(), "classes": list(CLASSES),
                       "in_channels": 6, "encoder": "resnet18"}
            payload.update({key: metadata[key] for key in ("steps", "seed") if key in metadata})
            torch.save(payload, root / "assets/model/model.pt")
        else:
            shutil.copyfile(checkpoint, root / "assets/model/model.pt")
        (root / "assets/config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (root / "predict.ipynb").write_text(json.dumps(make_notebook(architecture, stability="stability" in config),
                                           ensure_ascii=False, indent=2), encoding="utf-8")
        dependencies = ["torch>=2.5", "segmentation-models-pytorch==0.5.0", "numpy>=2.0",
                        "pillow>=10.0", "shapely>=2.0", "pyyaml>=6"]
        post = config.get("postprocess", {})
        settings = [post, *(post.get("classes") or {}).values(),
                    *(post.get(name, {}) for name in ("new_building", "tree_removal"))]
        if any(value.get("backend") == "rasterio" for value in settings):
            dependencies.append("rasterio>=1.4")
        if config.get("inference", {}).get("alignment", "none") != "none":
            dependencies.append("opencv-python-headless>=4.10,<6")
        (root / "requirements.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")
        for name in ("LICENSE", "NOTICE"):
            shutil.copyfile(baseline_root / name, root / name)
        with (root / "NOTICE").open("a", encoding="utf-8") as f:
            f.write("\nTerraDelta: modified offline inference, checkpoint validation and configurable postprocessing.\n"
                    "Original AIFactory baseline notices are retained. No competition performance claimed.\n")
        code = root / "assets/code/terradelta"
        for sub in ("models", "inference", "postprocess", "utils"):
            for source in (PACKAGE_ROOT / sub).glob("*.py"):
                if sub == "utils" and source.name not in {"__init__.py", "io.py"}:
                    continue
                if sub == "inference" and source.name in {"calibration.py", "v2_calibration.py", "v2_evaluation.py"}:
                    # Offline deployment needs inference only, not pilot cache/search.
                    continue
                if sub == "inference" and source.name == "stability_v23.py" and "stability" not in config:
                    continue
                target = code / sub / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        shutil.copyfile(PACKAGE_ROOT / "__init__.py", code / "__init__.py")
        # Rename only once all files have been successfully generated.
        root.rename(destination)
    return destination


def make_submission_zip(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    required = ["predict.ipynb", "requirements.txt", "LICENSE", "NOTICE", "assets/model/model.pt", "assets/config.yaml"]
    if not all((source / name).is_file() for name in required):
        raise ValueError("Incomplete submission export")
    if output.is_relative_to(source):
        raise ValueError("ZIP output must be outside source folder")
    if output.exists():
        raise FileExistsError(output)
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Submission symlinks are forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as f:
        temp = Path(f.name)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                    continue
                # Fixed metadata makes identical payloads reproducible across
                # exports, filesystems and source modification times.
                info = zipfile.ZipInfo(path.relative_to(source).as_posix(), (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with path.open("rb") as payload, archive.open(info, "w") as member:
                    shutil.copyfileobj(payload, member)
        if temp.stat().st_size > 6_000_000_000:
            raise ValueError("Submission ZIP exceeds competition 6 GB limit")
        temp.rename(output)
    finally:
        temp.unlink(missing_ok=True)
    return output
