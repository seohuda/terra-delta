"""Strict, safe checkpoint loading and atomic saves."""
from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

import torch

from terradelta import CLASSES


def load_checkpoint(path, model=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state_dict or be a plain state_dict")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    else:
        state, metadata = checkpoint, {}
    if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Invalid model state_dict")
    if "classes" in metadata and tuple(metadata["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order is incompatible")
    if metadata.get("in_channels", 6) != 6 or metadata.get("encoder", "resnet18") != "resnet18":
        raise ValueError("Checkpoint architecture is incompatible")
    if model is not None:
        model.load_state_dict(state, strict=True)
    return metadata


def save_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"classes": list(CLASSES), "in_channels": 6, "encoder": "resnet18", **metadata,
               "state_dict": model.state_dict()}
    fd, filename = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, filename)
        os.replace(filename, path)
    finally:
        Path(filename).unlink(missing_ok=True)
