#!/usr/bin/env python3
"""Bounded NAIP metadata discovery; raster download requires --download."""
import argparse
import json

from terradelta.external.common import DownloadGuard, download_assets, plan_summary
from terradelta.external.naip import discover_naip, estimate_aoi_bytes, sign_url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--region", default="")
    parser.add_argument("--state", help="Two-letter NAIP state code")
    parser.add_argument("--bands", choices=["rgb", "rgbnir", "nir"], default="rgb")
    parser.add_argument("--target-resolution", type=float, default=1.0)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--inspect-sizes", action="store_true", help="Sign transient URLs and HEAD assets; no payload")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: metadata only")
    mode.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", default="data/raw/naip")
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--max-file-bytes", type=int, default=50_000_000)
    parser.add_argument("--max-total-bytes", type=int, default=100_000_000)
    args = parser.parse_args(argv)
    result = discover_naip(bounds=args.bounds, years=args.years, region=args.region, state=args.state,
                           max_items=args.max_items, inspect_sizes=args.inspect_sizes or args.download)
    report = plan_summary(result["assets"])
    report.update(truncated=result["truncated"], region=args.region, bounds=args.bounds, years=args.years,
                  selected_bands=args.bands, target_resolution_m=args.target_resolution,
                  estimated_aoi_raw_bytes=estimate_aoi_bytes(args.bounds, target_resolution=args.target_resolution,
                                                            bands=args.bands, temporal_images=len(args.years)),
                  band_note="Downloads preserve all source COG bands; select/reproject bands in build_dataset tiles")
    print(json.dumps(report, indent=2))
    if args.download:
        if result["truncated"]:
            parser.error("Discovery truncated; narrow AOI/year before downloading")
        paths = download_assets(result["assets"], args.output_dir,
                                guard=DownloadGuard(True, args.max_files, args.max_total_bytes, args.max_file_bytes),
                                url_resolver=sign_url)
        print(json.dumps({"downloaded": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
