#!/usr/bin/env python3
"""Cache frozen CPU probabilities once, reproduce baseline, calibrate and resume."""
import argparse
from pathlib import Path
from terradelta.inference.calibration import analysis_artifacts, cache_validation, run_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline-score', type=float, required=True)
    a = p.parse_args()
    cache = Path(a.output) / 'probability-cache'
    cache_validation(a.manifest, a.checkpoint, cache)
    engine, result = run_sweep(cache, a.manifest, a.output, a.baseline_score)
    analysis_artifacts(engine, result, a.manifest)


if __name__ == '__main__':
    main()
