#!/usr/bin/env python3
"""Validate an existing local checkpoint through exported competition polygons."""

import argparse
from copy import deepcopy
import json

from terradelta.models.checkpoint import load_checkpoint
from terradelta.models.factory import build_model
from terradelta.training.splits import prepare_datasets
from terradelta.training.validation import validate_model
from terradelta.utils.io import atomic_json, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Local checkpoint; otherwise validation.checkpoint in config")
    parser.add_argument("--manifest", help="Explicit validation manifest; bypass configured split selection")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pred-dir", help="Save each sample's (3,H,W) probabilities as <id>.npy")
    parser.add_argument("--output", default="outputs/validation.json")
    args = parser.parse_args(argv)
    config = deepcopy(load_config(args.config))
    checkpoint = args.checkpoint or config.get("validation", {}).get("checkpoint")
    if not checkpoint:
        parser.error("Supply --checkpoint or validation.checkpoint; validation requires existing weights")
    # Existing checkpoints provide all weights; never fetch encoder weights.
    config.setdefault("model", {})["encoder_weights"] = None
    if args.pred_dir:
        config.setdefault("validation", {})["pred_dir"] = args.pred_dir
    if args.manifest:
        from terradelta.data.dataset import ChangeDataset

        dataset = ChangeDataset(args.manifest, transform=None, require_masks=True)
    else:
        _, dataset = prepare_datasets(config)
    model = build_model(config)
    load_checkpoint(checkpoint, model)
    scores = validate_model(model, dataset, config, device=args.device)
    atomic_json(args.output, scores)
    print(json.dumps(scores, indent=2, allow_nan=False))
    return scores


if __name__ == "__main__":
    main()
