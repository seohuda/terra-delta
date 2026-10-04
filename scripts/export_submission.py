#!/usr/bin/env python3
import argparse
from terradelta.submission import export_submission


def main():
    p = argparse.ArgumentParser(description="Export offline assets only; never submits")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/inference_baseline.yaml")
    p.add_argument("--output", default="outputs/submission_export")
    p.add_argument("--baseline-root")
    a = p.parse_args()
    print(export_submission(a.checkpoint, a.config, a.output, a.baseline_root))


if __name__ == "__main__":
    main()
