#!/usr/bin/env python3
"""Offline real-pilot review candidates from an explicit downloaded-source plan.

No download, training, synthetic data or label approval. RGB/NIR and static
footprints/spectral loss are review evidence. No final GT masks are emitted.
"""
import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds, Window
from shapely.geometry import box, shape, mapping
from shapely.strtree import STRtree

from terradelta.data.forest_labels import ForestLossRefiner
from terradelta.data.pairing import iter_pair_tiles, write_manifest
from terradelta.inference.alignment import estimate_translation
from terradelta.external.hansen import discover_hansen
from terradelta.utils.geo import rasterize_geometries, transform_geometry


def local_footprints(path, bounds, output, *, max_features=10000):
    """Stream a downloaded Microsoft GeoJSONL gzip, retain only AOI polygons."""
    aoi = box(*bounds)
    selected = []
    with gzip.open(path, "rt") as handle:
        for index, line in enumerate(handle):
            if len(line) > 1_000_000:
                raise ValueError("Footprint record exceeds bounded size")
            feature = json.loads(line)
            geometry = shape(feature["geometry"])
            if not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                continue
            if geometry.intersects(aoi):
                selected.append({"type": "Feature", "id": f"{Path(path).stem}_{index}",
                                 "properties": {"source_line": index}, "geometry": mapping(geometry)})
                if len(selected) > max_features:
                    raise ValueError("AOI footprints exceed feature limit")
    output.write_text(json.dumps({"type": "FeatureCollection", "features": selected}))
    return selected


def loss_window(dataset, tile):
    """Read a tiny local 30m window and project a mining aid, never GT."""
    extent = transform_bounds(tile["crs"], dataset.crs, *tile["bounds"])
    w = from_bounds(*extent, transform=dataset.transform)
    c, r = int(np.floor(w.col_off)) - 1, int(np.floor(w.row_off)) - 1
    w = Window(c, r, int(np.ceil(w.width)) + 3, int(np.ceil(w.height)) + 3)
    raw = dataset.read(1, window=w, boundless=True, fill_value=0)
    projected = np.zeros(tile["valid"].shape, np.uint8)
    reproject(raw, projected, src_transform=dataset.window_transform(w), src_crs=dataset.crs,
              dst_transform=tile["transform"], dst_crs=tile["crs"], resampling=Resampling.nearest)
    return projected


def vegetation(image):
    red, nir = image[..., 0].astype(np.float32), image[..., 3].astype(np.float32)
    ndvi = (nir - red) / np.maximum(nir + red, 1)
    return ndvi, ((ndvi + 1) / 2).astype(np.float32)


def build(root, output):
    if output.exists():
        raise FileExistsError("Use a fresh candidate directory; review evidence is preserved")
    plan = json.loads((root / "metadata/acquisition-plan.json").read_text())
    downloads = json.loads((root / "metadata/download-report.json").read_text())
    if not downloads.get("completed_at_utc"):
        raise ValueError("Source acquisition is incomplete")
    sources = {a["id"]: a for a in downloads["assets"]}
    ms = json.loads((root / "metadata/microsoft-selected.json").read_text())
    output.mkdir(parents=True)
    rows = []
    for region in plan["naip"]:
        name = region["region"]
        assets = sorted(region["assets"], key=lambda a: a["year"])
        pre, post = (sources[a["id"]] for a in assets)
        q = ms["wanted_quadkeys"][name]
        footprints = local_footprints(sources["microsoft_" + q]["path"], region["bounds"],
                                      output / f"{name}_footprints.geojson")
        with rasterio.open(pre["path"]) as ds:
            crs = ds.crs.to_string()
        projected = [shape(transform_geometry(f["geometry"], "EPSG:4326", crs)) for f in footprints]
        tree = STRtree(projected)
        loss_asset = sources[discover_hansen(bounds=region["bounds"], max_tiles=1)[0].id]
        overview = {}
        count = 0
        with ExitStack() as stack:
            loss = stack.enter_context(rasterio.open(loss_asset["path"]))
            for tile in iter_pair_tiles(pre["path"], post["path"], resolution=.6,
                    bounds=region["bounds"], bounds_crs="EPSG:4326", bands=(1, 2, 3, 4), max_tiles=500):
                index = f"r{tile['row']//256:02d}_c{tile['col']//256:02d}"
                identifier = f"{name}_{index}"
                folder = output / identifier
                folder.mkdir()
                before, after = tile["pre"], tile["post"]
                pn, ps = vegetation(before)
                qn, qs = vegetation(after)
                coarse = loss_window(loss, tile)
                candidate = (coarse > pre["year"] - 2000) & (coarse <= post["year"] - 2000)
                refined = ForestLossRefiner(pre_threshold=.65, post_threshold=.60, min_drop=.125).refine(
                    before[..., :3], after[..., :3], candidate, valid_mask=tile["valid"],
                    pre_vegetation=ps, post_vegetation=qs)
                reference = np.zeros((256, 256), np.uint8)
                proposed = np.zeros_like(reference)
                for i in tree.query(box(*tile["bounds"]), predicate="intersects"):
                    binary = rasterize_geometries([projected[int(i)]], out_shape=(256, 256),
                            transform=tile["transform"], src_crs=crs, dst_crs=crs).astype(bool)
                    reference |= binary
                    if binary.sum() < 40:
                        continue
                    difference = np.abs(before[..., :3].astype(float) - after[..., :3]).mean(axis=2)
                    if pn[binary].mean() > .15 and qn[binary].mean() < .2 and difference[binary].mean() > 25:
                        proposed |= binary
                for role, image in (("pre", before), ("post", after)):
                    Image.fromarray(image[..., :3]).save(folder / f"{role}.png")
                    Image.fromarray(image[..., 3]).save(folder / f"{role}_nir.png")
                for key, binary in (("building_footprint_reference", reference),
                                    ("proposed_new_building", proposed),
                                    ("hansen_candidate", candidate),
                                    ("proposed_tree_removal", refined["proposed_mask"]),
                                    ("valid_mask", tile["valid"])):
                    Image.fromarray(binary.astype(np.uint8) * 255).save(folder / f"{key}.png")
                alignment = estimate_translation(before[..., :3], after[..., :3])
                acquired = {a["id"]: a["acquired"] for a in region["raster_headers"]}
                row = {"id": identifier, "pre": str(folder / "pre.png"), "post": str(folder / "post.png"),
                       "new_building": "", "tree_removal": "", "label_status": "candidate",
                       "confidence": "UNREVIEWED", "audit_training_eligible": False,
                       "region_id": name, "state": name[:2], "source": "NAIP", "synthetic": False,
                       "year_pre": pre["year"], "year_post": post["year"],
                       "acquired_pre": acquired[pre["id"]], "acquired_post": acquired[post["id"]],
                       "resolution_m": .6, "bounds": json.dumps(tile["bounds"]), "crs": tile["crs"],
                       "coordinates": json.dumps(transform_bounds(tile["crs"], "EPSG:4326", *tile["bounds"])),
                       "transform": json.dumps(list(tile["transform"])[:6]),
                       "pre_source_raster": pre["id"], "post_source_raster": post["id"],
                       "source_raster": pre["id"] + "|" + post["id"],
                       "pre_source": pre["url"], "post_source": post["url"],
                       "source_url": json.dumps([pre["url"], post["url"]]),
                       "license_status": "unknown", "candidate_source": loss_asset["id"],
                       "building_reference_source": "microsoft_" + q,
                       "proposed_building_pixels": int(proposed.sum()),
                       "proposed_tree_pixels": int(refined["proposed_mask"].sum()),
                       "hansen_candidate_pixels": int(candidate.sum()),
                       "mean_rgb_difference": float(np.abs(before[..., :3].astype(float) - after[..., :3]).mean()),
                       **alignment}
                (folder / "metadata.json").write_text(json.dumps(row, indent=2))
                rows.append(row)
                overview[tile["row"]//256, tile["col"]//256] = (identifier, before[..., :3], after[..., :3])
                count += 1
        height = (1 + max(r for r, c in overview)) * 96
        width = (1 + max(c for r, c in overview)) * 96
        canvas = Image.new("RGB", (width * 2, height), "black")
        draw = ImageDraw.Draw(canvas)
        for (r, c), (identifier, a, b) in overview.items():
            for j, image in enumerate((a, b)):
                canvas.paste(Image.fromarray(image).resize((96, 96)), (c*96 + j*width, r*96))
                draw.text((c*96 + j*width + 2, r*96 + 2), identifier.split(name+'_')[1], fill="white", stroke_width=1, stroke_fill="black")
        canvas.save(output / f"{name}_overview.jpg", quality=82)
        print(json.dumps({"region": name, "candidate_tiles": count, "reference_footprints": len(footprints)}), flush=True)
    write_manifest(rows, output / "candidates.csv")
    print(json.dumps({"total_candidates": len(rows), "approved_labels": 0, "training_executed": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(Path(args.data_root).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
