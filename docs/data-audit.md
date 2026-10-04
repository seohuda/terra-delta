# Pre-training data audit — 2026-10-04

**NOT READY.** The repository contains zero real training samples and four locally
invented mock pairs. The earlier 1,000-pair/524 MB figure was a capacity estimate,
not an acquired or prepared dataset. This audit started from verified commit
`037fb91`; source manifests/images/masks and original weights were preserved.

## Actual filesystem inventory

Scan scope: `/home/lee/terra-delta`, excluding Git/environment/cache directories,
generated audit results and archive expansion. No external machine or AWS storage
was accessed. No real dataset is configured at `data/train_manifest.csv`.

| Item | Verified value |
| --- | --- |
| Canonical manifest | `outputs/smoke/manifest.csv`, 4 rows, 984 bytes |
| Existing split manifests | `outputs/smoke/train.csv`, 3 rows; `val.csv`, 1 row |
| Images/masks | `outputs/smoke/input/images/mock_00` through `mock_03` |
| Unique samples / files | 4 samples / 16 PNGs; split references are not extra samples |
| Images + masks, logical file sizes | 1,422,929 bytes (1.36 MiB) |
| All three manifests | 2,100 bytes; dataset payload total 1,425,029 bytes |
| Average images + masks / sample | 355,732.25 bytes |
| Real source rasters / footprints | None |

`inventory.json` records each discovered manifest and sample folder. Other local
output space includes pre-existing model copies and submission ZIPs and is not
additional dataset imagery. The installed environment was reused.

## Provenance and actual licenses

The source is `synthetic_mock`. `scripts/smoke_test.py` uses NumPy seed 0, random
RGB values 70–159 and two 40×40 constant-color rectangular edits. Masks are
invented; these are neither NAIP crops nor FEMA/Hansen-derived labels. Region,
state and 2020/2022 acquisition years are fixture placeholders. Coordinates,
source raster, physical resolution and real acquisition/license evidence are
absent. All four audited rows are **unverified / training-ineligible** despite
original `commercial_ok`/`reviewed` strings. The audited manifest preserves
original fields and adds the verdict/reasons; the training gate rejects its
unverified rows.

`LICENSE_DATA.md` includes the locally generated fixtures and existing official
checkpoint. The original checkpoint's local LICENSE grants competition-purpose
use; its unchanged bundled notices are retained. NAIP/FEMA/Hansen downloader
metadata exists, but their imagery/geometry is **not used** in these samples.
No actually used external sample source is absent from the license inventory.
No LEVIR-CD or AIHub sample exists in the audited dataset.

## Class distribution and integrity

| Measure | Background | New building | Tree removal |
| --- | ---: | ---: | ---: |
| Pixels | 258,944 | 1,600 | 1,600 |
| Pixel ratio | 98.7793% | 0.6104% | 0.6104% |
| Positive samples | — | 1 | 1 |
| Positive components (8-connectivity) | — | 1 | 1 |
| Empty binary masks | — | 3 | 3 |

Background-only=2; building-only=1; tree-only=1; both=0. Each positive has 1,600
pixels. Two combined change masks are empty. Background dominance is strong but
measured only on fixtures.

All 16 PNGs are readable, pairs are uint8 RGB 256×256, masks match dimensions,
binary values are legal and classes do not overlap. No corrupted/wrong-channel,
nonfinite, dark/bright, blank, mislabeled-empty or duplicate-across-sample pair
was found. Two background samples have identical pre/post: informational,
consistent with mock no-change construction. **Two suspicious samples** (`mock_00`,
`mock_01`) have constant-color fill. Visual inspection confirms random texture
with hard rectangular edits and GT-matching edges; neither edit resembles real
building construction/tree removal. Real inpainting quality, shadows, seasonal
variation and hard negatives cannot be assessed from these fixtures.

## Split and temporal evidence

Existing train IDs: `mock_00, mock_01, mock_03`; validation: `mock_02`. IDs and
paths cover the dataset exactly. Four logical region groups are disjoint and no
cross-partition source/image/content duplicate or temporal-chain candidate was
found. **Actual spatial/temporal isolation is unverified** because every region,
year and location is invented and no real coordinates/source raster exists.
An empty leakage CSV is not a certificate of real geographic separation.

Canonical grouping now joins shared pre/post source rasters, AOI/location IDs,
regions, image paths, parent/pair groups and touching/buffered georeferenced
bounds. A 2019→2021 / 2021→2023 chain with a common location/raster remains one
group even when region IDs differ. Preserve canonical source identifiers and
bounds/CRS, and audit both manifests before real training.

## Alignment (analysis only)

Phase correlation estimates pre-to-post translation without warping any image.
Four estimates exceed the response threshold 0.05; no >4-pixel shift occurs.

| Shift magnitude, pixels | Value |
| --- | ---: |
| Median | 0.003459 |
| Mean | 0.012584 |
| p90 | 0.032468 |
| p95 | 0.037943 |
| p99 | 0.042323 |
| Max | 0.043418 |

These nearly zero mock shifts cannot establish registration quality on real
imagery. Flat/nonfinite or low-response pairs are reported separately rather
than counted as reliable zero shifts.

## Official baseline: approximate local metric

Existing official weights were loaded strictly, `steps=100`, encoder weights
initialization disabled, CPU inference with gradients disabled. No optimizer,
backward pass, fit or parameter update was used.

Existing validation contains **one background mock**. Approximate local metric:
overall=0.25, building=0.25, tree=0.25; per class TP=FP=FN=0, TN=1. No-change FP
rate=0%; predicted area=0 and polygon count=0 for both classes. The 0.25 follows
local zero-denominator handling with no positive examples; it does not measure
positive-class detection.

Separate all-four-sample diagnostics: overall/class scores=0.214286; each class
TP=0, FP=0, FN=1, TN=3. No-change FP rate=0/2; average predicted area/polygon count
are zero. Missed examples are building `mock_00` and tree `mock_01`. Boundary
failures/class confusion/FP examples are absent because no positive polygon was
predicted. Shadow/season/registration causes are unassessable; none is invented.
All-four diagnostics include training-partition fixtures and are not held-out
validation. **Both numbers are approximate local metric values on invented
labels, not leaderboard predictions or a useful real A/B reference.**

## Readiness criteria and verdict

READY requires verified actual source/license/label-review records, usable
imagery/masks, a complete nonempty split and established geographic/temporal
separation. READY WITH WARNINGS permits reviewed data with measured nonblocking
imbalance/synthetic/registration concerns. NOT READY applies to mock-only data,
unverified rights/provenance, severe integrity failures or unestablished/leaking
geography. Current verdict: **NOT READY**, zero approved real samples.

Next work must happen on EC2 after separate AWS/data authorization: discover
bounded actual assets, resolve source-specific rights, acquire directly to EBS,
prepare/review labels and freeze disconnected regions. Re-run this audit and
unchanged-weight real validation before considering fine-tuning. First planned
checkpoints are **25/50/75/100/150/250**, using `configs/train_first250.yaml`; inspect
25–100 first. No 500+ step run is a priority. Compare per-class approximate scores,
FP/FN, no-change FP rate and confidence distributions with separate encoder/head
LRs. See [AWS workflow and exact commands](aws-training.md).

## Reproduction and small local artifacts

Using the already installed environment; no package installation is required:

```sh
.venv/bin/python scripts/audit_data.py \
  --manifest outputs/smoke/manifest.csv \
  --train-manifest outputs/smoke/train.csv --val-manifest outputs/smoke/val.csv \
  --checkpoint 'baseline/original/03_illegal structure submission/assets/model/unet_r18_cd.pt' \
  --output outputs/data_audit
```

Choose a fresh output directory for reruns. CLI forbids outgoing sockets and
preserves previous audit results. The script audits every sample, estimates
registration and runs existing-weight inference only. It does not train.

Local outputs under `outputs/data_audit/` (ignored by Git): `inventory.json`,
`class_distribution.json`, `sample_distribution.csv`, `sample_provenance.csv`,
`audited_manifest.csv`, `split_audit.json`, `spatial_leakage.csv`,
`temporal_leakage.csv`, `suspicious_samples.csv`, `alignment.csv/json`,
`synthetic_audit.json`, `duplicate_images.json`, `baseline_validation.json`,
`baseline_diagnostic.json`, `baseline_predictions.csv`, `error_analysis.json`,
`report.json`, and two JPEG montages. Preview JPEGs total **100,884 bytes**;
no absent category is fabricated and at most eight rows per montage are rendered.

NAIP/FEMA dry-runs replay existing bounded JSON plans via `--metadata-plan` with
AOI/year/provider checks; Hansen plans URLs locally without HEAD. Remote output
paths `/data/naip`, `/data/fema/structures.geojson`, `/data/hansen` are supported
and dry-runs do not create those directories. Cached sizes are historical and
must be rediscovered on EC2 before any authorized acquisition.

**Network:** dataset download=0 bytes; fresh external metadata/HEAD=0 bytes;
large transfers=none; environment/Docker pulls=none. Only small Git code/report
delivery uses the network, reported separately from dataset downloads. No AWS
start/create/stop, S3 transfer, GPU execution or competition submission occurred.

## Validation and disk accounting

Current code: **449 pytest tests passed**, zero failed/skipped; Ruff, Bash syntax,
git diff checks and original baseline file hashes passed. Full local test log:
`outputs/pytest-audit-final.txt`. Version-controlled small evidence summary:
[data-audit-summary.json](data-audit-summary.json).

Final audit directory: **125,412 bytes** including both montages,
metadata plans and reports. An initial audit snapshot (123,622 bytes) is
preserved locally in a temporary directory; both audit generations total
**249,034 bytes**. No data was downloaded. This counts
audit artifacts, not pre-existing model copies/ZIPs or temporary unit-test files.

Real manifests need explicit `provenance_status=verified`,
`license_review_status=verified`, `reviewer`, source/license evidence,
pre/post/mask sources, acquisition interval, resolution and geographic metadata.
These are documented human review decisions; the tool does not establish rights
merely from a URL or a status string.

## Git delivery

Verified audit implementation `5fb624eaa0577441bef820a1a604f9fd8d986851`
was published to main with exact local/remote SHA equality. Existing Git HTTPS
transport failures were avoided by uploading identical trees/commits through
GitHub Git API and advancing main without force. Only code and a small JSON/
Markdown summary are version controlled; source samples, weights and generated
audit images/reports are ignored. Final completion records are a subsequent
documentation commit. Git API traffic is small code delivery, separate from
the zero dataset-download bytes in the audit.
