# Offline-verifiable data pipeline

This stage implements and verifies preparation contracts. It does not train a
model, execute optimizer steps, call AWS or submit to AIFactory. Real acquisition
needs a later explicitly authorized bounded operation. The supported external
sources are **NAIP, FEMA USA Structures and Hansen**.

## Dataset and manifest contract

`ChangeDataset` reads a CSV, one sample directory, or a sorted root of directories.
Folder samples contain:

```text
sample-id/
  pre.png
  post.png
  new_building.png
  tree_removal.png
```

CSV requires unique nonempty safe `id`, `pre`, `post`. Paths resolve relative to the
CSV, including class and optional scope masks. `write_manifest` makes absolute
paths relative and retains extra provenance columns. Use the writer for real
paths rather than copying an example into a production manifest.

RGB arrays must have matching H/W; single-channel binary masks contain 0/1/255.
Overlapping class masks, malformed masks or missing required annotations raise.
Explicit reviewed absence is `absent` or `<class>_absent=true`; inference-only
loading with `require_masks=False` may return unspecified masks as background,
with `has_labels=False`. Never use that inference mode as ground-truth preparation.

Default output:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `id` | string | Manifest ID or directory name |
| `pre`, `post` | float32 `[3,H,W]` | Each timestamp independently normalized once |
| `image` | float32 `[6,H,W]` | pre RGB followed by post RGB |
| `mask` | int64 `[H,W]` | 0 background, 1 new building, 2 tree removal; `-100` outside labeled scope |
| `has_labels` | Boolean | Both classes supplied or explicitly marked absent |
| `valid_mask` | Boolean `[H,W]` | Pixels whose label is not ignored, after geometry |
| `auxiliary_mask` | optional Boolean `[2,H,W]` | Class 1 and class 2 channels with `return_auxiliary=True` |

ImageNet mean is `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, applied after
augmentation on values divided by 255. These are loader constants matching the
preserved baseline; they do not request pretrained weights or ImageNet images.

## Weak labels and partial scope

Optional manifest paths `valid_mask` and `review_mask` are binary masks on the
same high-resolution image grid. Trust their intersection; an omitted scope mask
defaults to all pixels. A supplied invalid/missing scope file raises. Class masks
must still obey their annotation contract. Outside scope, target pixels are
`-100`; the training losses ignore those pixels. An all-ignored target contributes
zero finite loss. No optimizer/backward operation is needed to test this behavior.

`PairedTransform` warps the signed int16 intermediate target with nearest-neighbor
interpolation, preserving `-100`; RGB uses linear interpolation. The returned
Boolean validity mask derives from the transformed target. Auxiliary class masks
are false outside scope, so callers must use validity to distinguish ignored
pixels from negatives. `has_labels=True` only establishes class annotation presence,
and can coexist with partial scope. Validation rejects partial scopes and needs
full ground truth; do not turn unreviewed forest pixels into background to satisfy it.

`ForestLossRefiner` proposes high-resolution vegetation-loss candidates but emits
no positive training target until an explicit manual mask/decision and reviewer
are supplied. Retain its `review_mask` and `valid_mask` when persisting a sample.
Its confidence is a heuristic, not a calibrated probability. A rejected candidate
is a reviewed negative only within reviewed scope. A real FEMA footprint is a
present-day building reference, not evidence that construction happened between
timestamps. Synthetic removal and independently reviewed temporal change are
different label provenances.

## Paired augmentation

Both timestamps and labels share flips, right-angle rotations and optional small
affine rotation/translation/scale. Photometric brightness, contrast, gamma, HSV,
blur, sharpen, JPEG and haze vary independently by timestamp. Rectangular inputs
avoid rotations that swap H/W. Local RNG seeds replay sequences, call-time seeds
replay a sample, and `reseed` supports distinct worker seeds. Evaluation uses
`training=False`. Geometry reflects borders consistently; masks use nearest values.

## Geographic and temporal partitioning

`terradelta.data.split` is the canonical split implementation, also used by the
training adapter. All strategies (`random`, `region`, `state`, `temporal`)
require `region_id`; even random splits assign connected geographic groups intact.
Shared pre/post source paths (including temporal role crossover), parent IDs,
pair IDs and explicit spatial groups join rows. When bounds are available, all
rows need bounds and explicit CRS. Touching/overlapping extents and buffered
neighbors also join after reprojection to a metric CRS. Without bounds, region
IDs must already encompass neighbors; the code cannot infer missing geography.

State splits additionally join entire states. Temporal validation requires
integer `temporal_year`, numeric `year_pre < year_post`, and holds out every connected group containing a `year_post`
at or above that cutoff, including older samples from those groups. Fewer than
two disconnected groups cannot produce a safe train/validation partition and raise.
`val_fraction` is an approximate sample target for indivisible groups, not an exact
guarantee. `assert_no_spatial_leak` rejects shared groups across partitions.

`training_eligibility` gates source and license status before dataset preparation.
See [the permission inventory](../LICENSE_DATA.md) for accepted status strings and
limitations: metadata assertions are not automatic permission/review verification.

## Bounded source discovery

Run from the repository with installed geo dependencies. These examples request
metadata or local plans only, without external image/geometry payloads:

```bash
.venv/bin/python scripts/download_naip.py --bounds -76.61 39.29 -76.60 39.30 \
  --years 2021 --state md --region md-smoke --max-items 1 --dry-run
.venv/bin/python scripts/download_fema.py --bounds -76.61 39.29 -76.60 39.30 --dry-run
.venv/bin/python scripts/download_hansen.py --bounds -76.61 39.29 -76.60 39.30 \
  --layers lossyear --max-tiles 1 --dry-run
```

NAIP uses [Microsoft's public STAC API](https://planetarycomputer.microsoft.com/docs/)
at `https://planetarycomputer.microsoft.com/api/stac/v1/search`, collection `naip`,
WGS84 bbox and acquisition datetime. State/year filters, deduplication and a
20-page bound restrict discovery. [SAS signing](https://planetarycomputer.microsoft.com/docs/concepts/sas/)
and HEAD are opt-in with size inspection; default dry-run does neither. Downloads
preserve full source COG bands; RGB/NIR selection and resolution belong to local tiling.

FEMA uses the [configured public polygon layer](https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0).
Dry-run reads layer metadata and `returnCountOnly=true`. Feature fetch is separately
opt-in, bounded by count and bytes, requests WGS84 GeoJSON with `outSR=4326`,
and orders/paginates by OBJECTID. Missing/duplicate IDs, wrong geometry, changing
counts or empty pages raise. Estimated bytes (`count * 1500`) are a planning
heuristic. Byte accounting also limits serialized responses/output, not an exact
HTTP transport bill. [Esri's official query contract](https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer/)
is the API reference.

Hansen URLs are pinned to [GFC-2024-v1.12](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html)
and 10° granules; only requested layers/AOI tiles are planned. The
[provider catalog](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12)
defines `lossyear=1..24` for 2001..2024, zero for no loss, and marks this version
superseded. Candidate selection uses `(year_pre, year_post]`, optional tree cover
and land datamask. Its roughly 30 m pixels cannot become 0.3–1 m segmentation
truth by upsampling. Review actual NAIP acquisition dates within a loss year.

API/schema details to preserve:

| Source | Required field/alias handling |
| --- | --- |
| NAIP | State filter reads `naip:state` (`md`, case-insensitive); resolution reads `gsd` with `naip:resolution` fallback. The verified item has `proj:shape` on properties and asset key `image`; no `file:size`, so HEAD is required for wire size. |
| FEMA | Service alias is `USA_Structures_View`, while layer 0 is named `USA_Structures_B`. The service uses Web Mercator WKID 102100 (EPSG:3857); AOI `inSR=4326` and GeoJSON `outSR=4326` are explicit, not an assertion that source polygons already use WGS84. |
| Hansen | Gate aliases `hansen`, `gfc`, `hansen_gfc` remain candidate-only; hyphens in source names normalize to underscores. Mixed provenance uses `+`, not comma-separated source names. The versioned download page's loss-year prose says 1–20 despite its 2024 coverage; the provider catalog and pinned code use 0–24. |

The training gate also excludes normalized `levir`, `levir_cd`, `levir_cd+`,
`aihub` and `ai_hub`. Preserve those recognizable names rather than using an alias
that bypasses review. The FEMA item declares attribution terms even though the
original discovery report labels it public domain; correct the status before
approving a derived sample.

## Download guards and estimates

NAIP/Hansen asset payloads need explicit `--download` and exact planned byte sizes,
usually from HEAD; uncompressed estimates never authorize a download. Default
limits are two files, 50,000,000 bytes per file and 100,000,000 total. Unsafe or
duplicate basenames, unknown sizes or preflight overages raise before creating
files or contacting payload URLs. HTTPS is required. Content-Length drift,
truncation or streaming overage aborts; reads go at most one byte past the bound.
Temporary `.part` files are cleaned and completed files publish without overwriting.
This is per-file atomicity: earlier completed files survive a later batch failure.
There is no silent retry or automatic download-size increase.

FEMA uses independent feature/byte caps (default 1,000 features and 10,000,000 bytes),
refuses existing output and does not silently accept an incomplete collection.
Production metadata reads default to 4,000,000 bytes per response; the live audit
used its own stricter 8,192-byte cap, not the production discovery defaults.

| Example | Planning quantity | Interpretation |
| --- | --- | --- |
| 256×256 RGB pre/post pair | 393,216 bytes before compression | Six uint8 channels, excluding masks/metadata |
| Same pair as normalized float32 | 1,572,864 bytes | In-memory image tensor, excluding runtime/model overhead |
| 1 km² at 1 m, RGB pre/post | ~6,000,000 raw bytes | Idealized AOI estimate, not full source COG wire size |
| Verified Maryland NAIP item shape 12,390×9,860×4 | 488,661,600 raw bytes; HEAD 447,829,958 wire bytes | Whole COG exceeds both default per-file and total guards |
| One Hansen 36,000×36,000 uint8 layer | 1,296,000,000 raw bytes | Full granule; `first`/`last` estimates use four channels |
| Verified Hansen 40N 080W lossyear HEAD | 42,217,473 planned wire bytes | One dated observation, no imagery read; other granules can differ |

No total real training download size is claimed without chosen AOIs/years,
complete discovery and HEAD inspection. Narrow the AOI or decline acquisition
when source assets exceed limits; do not substitute optimistic raw-AOI estimates.

The live size-inspection command also succeeded without acquiring imagery:

```bash
.venv/bin/python scripts/download_naip.py --bounds -76.61 39.29 -76.60 39.30 \
  --years 2021 --state md --region md-smoke --max-items 1 --dry-run --inspect-sizes
.venv/bin/python scripts/download_hansen.py --bounds -76.61 39.29 -76.60 39.30 \
  --layers lossyear --max-tiles 1 --dry-run --inspect-sizes
```

NAIP signed its asset URL transiently and used HEAD only. The local report
`outputs/naip-dry-run.json` records wire size **447,829,958 bytes**, raw estimate
**488,661,600 bytes** and a clean public URL for
`md_m_3907644_sw_18_060_20210617`. No TIFF GET occurred. A small AOI does not make
this whole COG a small acquisition. Hansen's corresponding report confirms
**42,217,473 wire bytes**, with no asset GET. The production CLI runs used the
normal 4,000,000-byte metadata limit per response, separately from the stricter
8/16 KiB audit probes; their byte totals must not be conflated.

The production FEMA metadata/count CLI also succeeded. Its local report
`outputs/fema-dry-run.json` records 218 features, 2,000 records per page,
`supports_pagination=true`, OBJECTID ordering and a 327,000-byte GeoJSON heuristic,
with no exact feature-body size and no geometry fetched. The final discovery
reports `license_status=unknown` while the stated CC BY 4.0 text and BY 3.0 badge
are reviewed; public access is not a training approval. See
[the item terms and discrepancy](licenses.md).

## Local CRS tiling and preparation

`discover_temporal_pairs` groups by region and filters year gaps. `iter_pair_tiles`
accepts existing local rasters only, requires CRS and uint8 bands, constructs a
common intersection grid projected in meters and uses bounded WarpedVRT windows.
AOI bounds need `bounds_crs`; resolution is meters per pixel. Complete 256×256
tiles are emitted by default, smaller edge strips discarded, with validity and
georeferencing. Grid/tiling limits reject unexpectedly large preparation jobs.
NIR is a fourth data band, not an alpha band; inspect raster color interpretation.
The tiler supports existing alpha for RGB validity and rejects requested image
bands tagged as alpha. It does not silently interpret an alpha-tagged fourth band
as NIR or a NIR band as coverage. No real raster was fetched to verify provider
band metadata on disk; local GeoTIFF regression tests establish band handling.

FEMA rasterization explicitly transforms GeoJSON to raster CRS, uses pixel-edge
affines, supports polygon holes and rejects oversized pixel grids. Footprints
are auxiliary references, never automatically temporal labels.

Examples for **already-local** inputs, still dry-run:

```bash
.venv/bin/python scripts/build_dataset.py tiles \
  --pre data/raw/naip/pre.tif --post data/raw/naip/post.tif \
  --region-id md-region-a --state md --source NAIP --license-status unknown \
  --year-pre 2019 --year-post 2021 --target-resolution 1 --tile-size 256 \
  --max-tiles 4 --output-dir data/candidates/md-region-a --dry-run
.venv/bin/python scripts/build_dataset.py manifest \
  --input-manifest data/manifests/reviewed.csv --split region --buffer-m 30 \
  --output-dir data/manifests/split --dry-run
```

Tiling dry-run reads local windows but writes no imagery/manifests. Actual tiling
outputs `candidates.csv` with `label_status=candidate`; it does not fabricate class
masks or approve unknown licenses. A temporal manifest split additionally needs
`--temporal-year YEAR`. Persist reviewed class masks and scope/provenance before
creating training manifests; split configuration and dataset sizes belong to the
training integration.

## Verification status

The [source audit](licenses.md) records exact live metadata endpoints, caps,
response sizes and permissions. Live verification covers NAIP discovery, transient
SAS and signed HEAD, FEMA metadata/count/pagination capability and a pinned Hansen
HEAD. Production metadata-only dry-runs succeeded for all three sources. FEMA
geometry fetch/pagination, real raster tiling and final label quality remain
unverified; advertised pagination support is not a fetched-page validation.

Run the bounded offline tests:

```bash
.venv/bin/python -m pytest -q tests/test_dataset.py tests/test_external.py tests/test_synthetic.py
```

They generate tiny fixtures and mock HTTP; they cover annotations/normalization,
relative manifests, review-scope/ignore-label behavior, augmentation, canonical
geographic/temporal grouping, reprojection, holes/NIR/nodata, discovery filters,
source gates and streaming failure cleanup. Loss checks evaluate tensors only;
validation rejection uses a mock predictor. No training loop/optimizer runs.

Two source regressions are covered:
`augmentation.enabled=False` now preserves inputs, and final-page NAIP discovery
reports truncation when its item cap omits assets. Regression tests remain active
without skip/xfail markers.

Final bounded rerun on 2026-10-04: **165 passed**, with 47 existing affine/rasterio
pending-deprecation warnings; no skipped/xfail regressions. Ruff passed for both
assigned test files. This is the three-file suite shown above, not a claim that
the complete repository suite or real-data acquisition was verified here.
