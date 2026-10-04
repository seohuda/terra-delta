#!/usr/bin/env python3
"""Tile local georasters or build license-gated, spatially disjoint manifests."""
import argparse
import json
from pathlib import Path

from PIL import Image

from terradelta.data.dataset import ChangeDataset, read_manifest
from terradelta.data.pairing import filter_training_rows, iter_pair_tiles, write_manifest
from terradelta.data.split import assert_no_spatial_leak, split_manifest
from terradelta.external.fema_structures import rasterize_buildings


def _tiles(args):
    if args.year_pre >= args.year_post:
        raise ValueError("Temporal pair requires year_pre < year_post")
    footprints = json.loads(Path(args.footprints).read_text()) if args.footprints else None
    rows = []
    output = Path(args.output_dir).resolve()
    for tile in iter_pair_tiles(args.pre, args.post, resolution=args.target_resolution,
                                dst_crs=args.crs, bounds=args.bounds, bounds_crs=args.bounds_crs,
                                tile_size=args.tile_size, bands=(1, 2, 3, 4) if args.bands == "rgbnir" else (1, 2, 3),
                                max_tiles=args.max_tiles):
        identifier = f"{args.region_id}_{args.year_pre}_{args.year_post}_{tile['row']}_{tile['col']}"
        from terradelta.external.common import safe_filename
        identifier = safe_filename(identifier)
        folder = output / identifier
        row = {"id": identifier, "pre": str(folder / "pre.png"), "post": str(folder / "post.png"),
               "new_building": "", "tree_removal": "", "region_id": args.region_id,
               "state": args.state, "source": args.source, "year_pre": args.year_pre, "year_post": args.year_post,
               "license_status": args.license_status, "label_status": "candidate",
               "bounds": json.dumps(tile["bounds"]), "crs": tile["crs"],
               "transform": json.dumps(list(tile["transform"])[:6]), "resolution_m": args.target_resolution,
               "pair_id": f"{Path(args.pre).resolve()}|{Path(args.post).resolve()}"}
        if not args.dry_run:
            folder.mkdir(parents=True, exist_ok=False)
            for key in ("pre", "post"):
                Image.fromarray(tile[key][..., :3]).save(folder / f"{key}.png")
                if args.bands == "rgbnir":
                    Image.fromarray(tile[key][..., 3]).save(folder / f"{key}_nir.png")
            if footprints is not None:
                mask = rasterize_buildings(footprints, transform=tile["transform"], out_shape=tile["valid"].shape,
                                           raster_crs=tile["crs"])
                Image.fromarray(mask * 255).save(folder / "building_footprint_reference.png")
        rows.append(row)
    if not rows:
        raise ValueError("No complete valid tiles; inspect coverage/resolution and AOI")
    if not args.dry_run:
        write_manifest(rows, output / "candidates.csv")
    print(json.dumps({"dry_run": args.dry_run, "tiles": len(rows), "label_status": "candidate",
                      "training_eligible": False, "samples": rows}, indent=2))


def _manifest(args):
    if args.input_manifest:
        rows = read_manifest(args.input_manifest)
    else:
        samples = ChangeDataset(args.samples_dir).samples
        rows = [{**row, "region_id": args.region_id, "state": args.state, "source": args.source,
                 "year_pre": args.year_pre, "year_post": args.year_post, "license_status": args.license_status}
                for row in samples]
    eligible, excluded = filter_training_rows(rows)
    output = Path(args.output_dir)
    report = {"dry_run": args.dry_run, "eligible": len(eligible), "excluded": len(excluded),
              "exclusions": [{"id": row["id"], "reason": row["exclusion_reason"]} for row in excluded]}
    if not eligible:
        if not args.dry_run:
            write_manifest(excluded, output / "excluded.csv")
        print(json.dumps(report, indent=2))
        raise ValueError("No commercially approved, reviewed training rows")
    parts = split_manifest(eligible, strategy=args.split, val_fraction=args.val_fraction, seed=args.seed,
                           buffer_m=args.buffer_m, temporal_year=args.temporal_year)
    assert_no_spatial_leak(parts["train"], parts["val"], buffer_m=args.buffer_m)
    report.update(train=len(parts["train"]), val=len(parts["val"]))
    if not args.dry_run:
        # Validation is loader-only: no forward, optimizer or training operation.
        for row in eligible:
            for key in ("pre", "post"):
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
        for name, partition in parts.items():
            path = output / f"{name}.csv"
            write_manifest(partition, path)
            dataset = ChangeDataset(path)
            for i in range(len(dataset)):
                dataset[i]
        write_manifest(excluded, output / "excluded.csv")
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    tile = subs.add_parser("tiles", help="Local CRS-aware temporal tiling; output remains unlabeled candidates")
    tile.add_argument("--pre", required=True)
    tile.add_argument("--post", required=True)
    tile.add_argument("--footprints", help="WGS84 FEMA GeoJSON: reference footprints, NOT temporal GT")
    tile.add_argument("--crs", help="Projected target CRS in meters; defaults to pre raster CRS")
    tile.add_argument("--bounds", type=float, nargs=4)
    tile.add_argument("--bounds-crs")
    tile.add_argument("--target-resolution", type=float, default=1.0)
    tile.add_argument("--bands", choices=["rgb", "rgbnir"], default="rgb")
    tile.add_argument("--tile-size", type=int, default=256)
    tile.add_argument("--max-tiles", type=int, default=100)
    tile.set_defaults(action=_tiles)
    manifest = subs.add_parser("manifest", help="Validate labels, gate licenses, and split reviewed local data")
    input_group = manifest.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-manifest")
    input_group.add_argument("--samples-dir")
    manifest.add_argument("--split", choices=["random", "region", "state", "temporal"], default="region")
    manifest.add_argument("--val-fraction", type=float, default=0.2)
    manifest.add_argument("--buffer-m", type=float, default=0.0)
    manifest.add_argument("--temporal-year", type=int)
    manifest.add_argument("--seed", type=int, default=0)
    manifest.set_defaults(action=_manifest)
    for sub in (tile, manifest):
        sub.add_argument("--output-dir", required=True)
        sub.add_argument("--region-id", default="")
        sub.add_argument("--state", default="")
        sub.add_argument("--source", default="unknown")
        sub.add_argument("--license-status", default="unknown")
        sub.add_argument("--year-pre", type=int, default=2020)
        sub.add_argument("--year-post", type=int, default=2022)
        sub.add_argument("--dry-run", action="store_true", help="Plan local outputs without writing images/manifests")
    args = parser.parse_args(argv)
    if args.command == "tiles" and not args.region_id:
        parser.error("tiles requires --region-id")
    args.action(args)


if __name__ == "__main__":
    main()
