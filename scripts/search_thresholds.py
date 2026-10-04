#!/usr/bin/env python3
"""Tune polygon postprocessing on saved probabilities without running a model."""

import argparse
from copy import deepcopy
import json

from terradelta.training.search import CLASSES, DEFAULT_GRID, search_thresholds
from terradelta.utils.io import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, help="Directory of <id>.npy files with shape (3,H,W)")
    parser.add_argument("--ground-truth", required=True, help="Polygon GT CSV or already-split labeled manifest CSV")
    parser.add_argument("--ground-truth-kind", choices=("auto", "polygons", "manifest"), default="auto")
    parser.add_argument("--config", help="Optional YAML; preserves inference and other postprocessing settings")
    parser.add_argument("--objective", help="Evaluator scalar key, e.g. score or classes.new_building.score")
    parser.add_argument("--method", "--mode", choices=("staged", "cartesian"))
    parser.add_argument("--minimize", action="store_true", help="Minimize the selected evaluator score")
    parser.add_argument("--output-csv", default="outputs/threshold_search.csv")
    parser.add_argument("--best-yaml", default="outputs/best_postprocess.yaml")
    flags = {"threshold": "thresholds", "min_area": "min-areas",
             "min_pos_area": "min-pos-areas", "simplify_px": "simplify-px"}
    for prefix in ("building", "tree"):
        for parameter, suffix in flags.items():
            parser.add_argument(f"--{prefix}-{suffix}", type=float, nargs="+",
                                dest=f"{prefix}_{parameter}")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else {}
    options = config.get("search", {})
    objective = args.objective or options.get("objective")
    if not objective:
        parser.error("Specify --objective or search.objective explicitly from the evaluator dictionary")
    grid = deepcopy(options.get("grid", DEFAULT_GRID))
    for name, prefix in zip(CLASSES, ("building", "tree")):
        for parameter in flags:
            values = getattr(args, f"{prefix}_{parameter}")
            if values is not None:
                grid.setdefault(name, {})[parameter] = values
    result = search_thresholds(args.pred_dir, args.ground_truth, config,
                              objective=objective, grid=grid, method=args.method or options.get("method", "staged"),
                              maximize=False if args.minimize else options.get("maximize", True),
                              ground_truth_kind=args.ground_truth_kind,
                              output_csv=args.output_csv, best_yaml=args.best_yaml)
    print(json.dumps({"objective": result["objective"], "best_score": result["best_score"],
                      "trials": len(result["results"]), "method": result["method"],
                      "output_csv": args.output_csv, "best_yaml": args.best_yaml}, indent=2))
    return result


if __name__ == "__main__":
    main()
