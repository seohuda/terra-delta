#!/usr/bin/env python3
"""Generate competition-format CSV and optional raw probability maps."""
import argparse

from terradelta.inference.predictor import predict_directory
from terradelta.utils.io import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/inference_baseline.yaml")
    p.add_argument("--output", default="outputs/prediction.csv")
    p.add_argument("--prob-dir")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    predict_directory(args.input_dir, args.output, args.checkpoint, load_config(args.config),
                      device=args.device, probability_dir=args.prob_dir)


if __name__ == "__main__":
    main()
