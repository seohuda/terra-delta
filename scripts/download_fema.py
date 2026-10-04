#!/usr/bin/env python3
"""USA Structures AOI count dry-run and explicitly bounded GeoJSON fetch."""
import argparse
import json

from terradelta.external.fema_structures import FEATURE_LAYER, discover_fema, fetch_fema


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", nargs=4, type=float, required=True)
    parser.add_argument("--endpoint", default=FEATURE_LAYER, help="Verified USA Structures FeatureServer layer")
    parser.add_argument("--output", default="data/raw/fema/structures.geojson")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: layer metadata and AOI count only")
    mode.add_argument("--download", action="store_true")
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--page-size", type=int, default=200)
    args = parser.parse_args(argv)
    report = discover_fema(bounds=args.bounds, endpoint=args.endpoint)
    print(json.dumps({**report, "dry_run": not args.download, "expected_file": args.output}, indent=2))
    if args.download:
        result = fetch_fema(bounds=args.bounds, endpoint=args.endpoint, output=args.output, allow_download=True,
                            max_features=args.max_features, max_bytes=args.max_bytes, page_size=args.page_size)
        print(json.dumps({"features": len(result["features"]), "file": args.output}))


if __name__ == "__main__":
    main()
