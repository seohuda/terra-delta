#!/usr/bin/env python3
"""Complete/reuse the existing v2 coarse sweep; never generate data or train."""

import argparse
from terradelta.inference.v2_calibration import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--legacy", required=True)
    p.add_argument("--stress", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(a.checkpoint_dir, a.legacy, a.stress, a.output, a.device)
