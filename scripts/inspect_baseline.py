#!/usr/bin/env python3
"""Inspect immutable baseline without executing any notebook cells."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from terradelta.models.factory import build_model
from terradelta.models.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("baseline/original/03_illegal structure submission"))
    parser.add_argument("--forward", action="store_true", help="One CPU forward; no training")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Baseline missing. See baseline/README.md for retrieval.")
    for path in sorted(args.root.rglob("*")):
        if path.is_file():
            print(path.relative_to(args.root), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    model = build_model().eval()
    metadata = load_checkpoint(args.root / "assets/model/unet_r18_cd.pt", model)
    print(json.dumps(metadata, indent=2))
    if args.forward:
        torch.set_num_threads(2)
        with torch.inference_mode():
            logits = model(torch.zeros(1, 6, 256, 256))
        print("forward", list(logits.shape), "finite", bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    main()
