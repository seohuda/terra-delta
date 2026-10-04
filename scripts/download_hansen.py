#!/usr/bin/env python3
"""Plan pinned Hansen granules for candidate mining, never direct GT."""
import argparse
import json

from terradelta.external.common import DownloadGuard, download_assets, plan_summary
from terradelta.external.hansen import LAYERS, VERSION, discover_hansen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", nargs=4, type=float, required=True)
    parser.add_argument("--layers", nargs="+", choices=sorted(LAYERS), default=["lossyear"])
    parser.add_argument("--max-tiles", type=int, default=8)
    parser.add_argument("--inspect-sizes", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: only asset URLs and estimates")
    mode.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", default="data/raw/hansen")
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--max-file-bytes", type=int, default=50_000_000)
    parser.add_argument("--max-total-bytes", type=int, default=100_000_000)
    args = parser.parse_args(argv)
    assets = discover_hansen(bounds=args.bounds, layers=args.layers, max_tiles=args.max_tiles,
                             inspect_sizes=args.inspect_sizes or args.download)
    print(json.dumps({**plan_summary(assets), "version": VERSION, "purpose": "candidate_mining_only"}, indent=2))
    if args.download:
        paths = download_assets(assets, args.output_dir,
                                guard=DownloadGuard(True, args.max_files, args.max_total_bytes, args.max_file_bytes))
        print(json.dumps({"downloaded": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
