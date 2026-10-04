#!/usr/bin/env python3
"""USA Structures AOI count dry-run and explicitly bounded GeoJSON fetch."""
import argparse
import json

from terradelta.external.fema_structures import FEATURE_LAYER, discover_fema, fetch_fema
from terradelta.external.common import cached_metadata_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", nargs=4, type=float, required=True)
    parser.add_argument("--endpoint", default=FEATURE_LAYER, help="Verified USA Structures FeatureServer layer")
    parser.add_argument("--output", default="data/raw/fema/structures.geojson")
    parser.add_argument("--metadata-plan", help="Replay an existing dry-run JSON offline; no network")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: layer metadata and AOI count only")
    mode.add_argument("--download", action="store_true")
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--page-size", type=int, default=200)
    args = parser.parse_args(argv)
    if args.metadata_plan:
        if args.download:
            parser.error("Cached plans are offline dry-run only; rediscover on EC2 before downloading")
        report = cached_metadata_plan(args.metadata_plan, source="FEMA_USA_Structures", bounds=args.bounds)
        if report.get("endpoint") != args.endpoint:
            parser.error("Cached endpoint differs from requested provider")
        report.update(expected_file=args.output, output_path=args.output)
        print(json.dumps(report, indent=2))
        return report
    report = discover_fema(bounds=args.bounds, endpoint=args.endpoint)
    print(json.dumps({**report, "dry_run": not args.download, "expected_file": args.output, "output_path": args.output}, indent=2))
    if args.download:
        result = fetch_fema(bounds=args.bounds, endpoint=args.endpoint, output=args.output, allow_download=True,
                            max_features=args.max_features, max_bytes=args.max_bytes, page_size=args.page_size)
        print(json.dumps({"features": len(result["features"]), "file": args.output}))


if __name__ == "__main__":
    main()
