#!/usr/bin/env python3
"""Bounded CPU polygon benchmark on synthetic masks; no model or training."""
import argparse
import json
from time import perf_counter

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from terradelta.postprocess.polygons import mask_to_polygons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("Use 1..20 repeats")
    rectangle = np.zeros((256,256), dtype=bool)
    rectangle[20:200,30:220] = True
    donut = rectangle.copy()
    donut[60:160,70:160] = False
    fragmented = np.random.default_rng(0).random((256,256)) > .92
    records = []
    for name, mask in (("empty", np.zeros_like(rectangle)), ("rectangle", rectangle),
                       ("donut", donut), ("fragmented", fragmented), ("full", np.ones_like(rectangle))):
        geometries = {}
        for backend in ("reference", "optimized"):
            start = perf_counter()
            for _ in range(args.repeats):
                output = mask_to_polygons(mask, backend=backend, simplify_px=0)
            seconds = (perf_counter()-start)/args.repeats
            geometries[backend] = unary_union([Polygon(p) for p in output])
            records.append({"mask":name, "backend":backend, "seconds":seconds, "polygons":len(output)})
        if not geometries["reference"].equals(geometries["optimized"]):
            raise AssertionError(f"Geometry mismatch: {name}")
    print(json.dumps(records,indent=2))


if __name__ == "__main__":
    main()
