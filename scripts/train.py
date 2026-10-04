#!/usr/bin/env python3
"""Prepare a training run; --dry-run performs only data checks and one forward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="No optimizer/scheduler construction and no training")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda or cuda:N (unavailable CUDA falls back to CPU)")
    parser.add_argument("--resume", help="Full TerraDelta training checkpoint")
    parser.add_argument("--init-checkpoint", help="Load strict baseline/model weights without resuming optimizer")
    args = parser.parse_args(argv)

    from terradelta.models.factory import build_model
    from terradelta.models.checkpoint import load_checkpoint
    from terradelta.training import Trainer, dry_run, prepare_datasets
    from terradelta.training.trainer import validate_training_config
    from terradelta.utils.seed import seed_everything
    from terradelta.utils.io import load_config, training_run_lock

    config = load_config(args.config)
    cfg = config.setdefault("training", {})
    if args.resume:
        cfg["resume"] = args.resume
        cfg["init_checkpoint"] = None
    if args.init_checkpoint:
        cfg["init_checkpoint"] = args.init_checkpoint
    if cfg.get("resume") and cfg.get("init_checkpoint"):
        parser.error("Choose resume or init_checkpoint, not both")
    validated = validate_training_config(config)
    # Offline construction: downloaded encoder weights are never fetched by this CLI.
    if config.get("model", {}).get("encoder_weights") is not None:
        parser.error("Use model.encoder_weights: null and a local init_checkpoint; this CLI never downloads weights")
    seed_everything(validated["seed"], validated["deterministic"])
    augmentation = config.get("data", {}).get("augmentation", {})
    transform = None
    if augmentation.get("enabled", False):
        from terradelta.data.transforms import build_transforms
        transform = build_transforms(augmentation, training=True, seed=validated["seed"])
    train, validation = prepare_datasets(config, train_transform=transform)
    model = build_model(config)
    initial_metadata = None
    if cfg.get("init_checkpoint"):
        initial_metadata = load_checkpoint(cfg["init_checkpoint"], model=model)
        # Metadata can include tensors/optimizer state in training checkpoints.
        initial_metadata = {key: initial_metadata[key] for key in ("classes", "in_channels", "encoder", "steps", "seed") if key in initial_metadata}
    elif args.dry_run and cfg.get("resume"):
        metadata = load_checkpoint(cfg["resume"], model=model)
        if metadata.get("training_format_version") != 1:
            parser.error("Resume needs a TerraDelta training checkpoint; use --init-checkpoint for baseline weights")
        initial_metadata = {key: metadata[key] for key in ("steps", "seed", "training_format_version") if key in metadata}
    device = args.device or cfg.get("device", "auto")
    if args.dry_run:
        plan = dry_run(model, train, validation, config, device=device)
        plan["initial_checkpoint_metadata"] = initial_metadata
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    if initial_metadata is not None:
        config["training"]["initial_checkpoint_metadata"] = initial_metadata
    with training_run_lock(validated["output_dir"], resume=bool(cfg.get("resume"))):
        trainer = Trainer(model, train, validation, config, device=device)
        print(json.dumps(trainer.fit(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
