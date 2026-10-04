#!/usr/bin/env python3
"""Bounded NAIP metadata discovery; raster download requires --download."""
import argparse
import json

from terradelta.external.common import DownloadGuard, cached_metadata_plan, download_assets, plan_summary
from terradelta.external.naip import discover_naip, discover_naip_years, estimate_aoi_bytes, sign_url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    temporal = parser.add_mutually_exclusive_group(required=True)
    temporal.add_argument("--years", type=int, nargs="+")
    temporal.add_argument("--list-years", action="store_true", help="Bounded acquisition-year discovery; metadata only")
    parser.add_argument("--region", default="")
    parser.add_argument("--state", help="Two-letter NAIP state code")
    parser.add_argument("--bands", choices=["rgb", "rgbnir", "nir"], default="rgb")
    parser.add_argument("--target-resolution", type=float, default=1.0)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--inspect-sizes", action="store_true", help="Sign transient URLs and HEAD assets; no payload")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: metadata only")
    mode.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", "--output", default="data/raw/naip")
    parser.add_argument("--metadata-plan", help="Replay an existing dry-run JSON offline; no network")
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--max-file-bytes", type=int, default=50_000_000)
    parser.add_argument("--max-total-bytes", type=int, default=100_000_000)
    args = parser.parse_args(argv)
    if args.list_years:
        if args.download or args.inspect_sizes or args.metadata_plan:
            parser.error("--list-years is live metadata only; no downloads/HEAD/cached single-year plan")
        report = discover_naip_years(bounds=args.bounds, region=args.region, state=args.state, max_items=args.max_items)
        report["output_path"] = args.output_dir
        print(json.dumps(report, indent=2))
        return report
    if args.metadata_plan:
        if args.download or args.inspect_sizes:
            parser.error("Cached plans are offline dry-run only; rediscover on EC2 before downloading")
        report = cached_metadata_plan(args.metadata_plan, source="NAIP", bounds=args.bounds, years=args.years)
        report.update(output_path=args.output_dir, selected_bands=args.bands, target_resolution_m=args.target_resolution)
        print(json.dumps(report, indent=2))
        return report
    result = discover_naip(bounds=args.bounds, years=args.years, region=args.region, state=args.state,
                           max_items=args.max_items, inspect_sizes=args.inspect_sizes or args.download)
    report = plan_summary(result["assets"])
    report.update(source="NAIP", output_path=args.output_dir, truncated=result["truncated"], region=args.region, bounds=args.bounds, years=args.years,
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
