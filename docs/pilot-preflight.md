# Real pilot: acquisition blocked, preparation verified

Historical preflight before the user authorized a dedicated CPU builder.
That host-access blocker was subsequently resolved. See the
[actual acquisition, audit and stopped-instance report](real-pilot-report.md).
The status table below preserves the earlier zero-acquisition checkpoint.

Status on 2026-10-04: **NOT READY**. No real pilot pair has been acquired or
reviewed. The approved existing EC2 instance ID, region and SSH target are
missing; an optional existing S3 bucket is also unspecified. The default
`ap-northeast-2` read-only lookup for `Project=TerraDelta` or `terra-delta`
returned no instances. That is not an inventory of every region or an approval
to use another project's host.

Local preparation adds bounded NAIP acquisition-year discovery. It uses actual
STAC dates and only proposes 2–4-year combinations when an individual footprint
covers the requested AOI in each year. Intersection alone does not qualify.
Truncation is explicit and cannot establish absence of other years. This is
metadata coverage, not verification of valid pixels or actual change.

## Source review before acquisition

These are candidate sources, not sources already used to produce a real pilot.
Save the exact asset URL/ID, dates, bounds, resolution, license evidence and
attribution with each selected source on EC2 before accepting derived labels.

| Source | Reviewed terms / applicability | Allowed pilot role |
| --- | --- | --- |
| USDA NAIP via Planetary Computer | [USDA catalog](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) identifies public-domain imagery. Verify selected assets and acquisition dates. | RGB pre/post imagery; optional NIR for label review only. |
| NAIP AWS buckets | [AWS registry](https://registry.opendata.aws/naip/) lists public-domain terms with attribution, **Requester Pays**, `us-west-2`. | Not selected. Do not assume anonymous access or enable requester billing. Prefer the existing Planetary Computer path directly from EC2. |
| Hansen GFC 2024 v1.12 | [Versioned download page](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html): CC BY 4.0 with attribution; approximately 30 m. | Loss-year candidate mining only; never directly resize into ground truth. |
| FEMA USA Structures item | [Item page](https://www.arcgis.com/home/item.html?id=0ec8512ad21e4bb987d7e848d14e7e24). Current license evidence remains unresolved; previous metadata had inconsistent license text/badge. | Excluded until exact license evidence is resolved. Static footprints cannot prove construction between dates. |
| Microsoft Global ML Building Footprints | [Repository](https://github.com/microsoft/GlobalMLBuildingFootprints), [license](https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/LICENSE): CDLA Permissive 2.0; retain the agreement when sharing data. Distinct from Microsoft US Building Footprints under ODbL. | Possible static candidate/reference geometry only, with independent NAIP temporal review. Not downloaded. |
| Google Open Buildings Temporal v1 | [Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1): CC BY 4.0 or ODbL; effective resolution about 4 m. [Coverage](https://sites.research.google/gr/open-buildings/temporal/) does not include continental US. | Excluded from a CONUS NAIP pilot. No download merely to satisfy a source list. |

## First steps after the host is identified

Run on the approved existing running EC2. Do not start/create an instance or
bucket. Do not install environments, download source payloads, or transfer
weights through the hotspot PC. Data flow is provider → EC2 → EBS/approved S3.

```sh
# EC2, existing checkout: first verify ancestry and storage.
git pull --ff-only
git merge-base --is-ancestor 7fc9f8996b1232b6944c7f624d386d41a2ad103d HEAD
df -h /data
du -sh /data/terradelta
```

If `/data/terradelta` does not yet exist, record that rather than treating a
failed `du` as a size measurement. Inspect mount, write access and available
space before setup. The existing setup script requires CUDA; a CPU-only data
host needs that requirement reviewed before executing it. No environment setup
has been attempted in this phase.

For each small candidate AOI, list actual years **on EC2** before choosing dates.
The following bounds are a planning probe, not an acquired or accepted region:

```sh
.venv/bin/python scripts/download_naip.py \
  --bounds -76.61 39.29 -76.60 39.30 --state md --region md-probe \
  --list-years --max-items 200 --dry-run \
  --output /data/terradelta/raw/naip
```

This command reads bounded STAC metadata, with no signing, HEAD, raster reads,
payload download, or output-directory creation. No such live AOI query was run
locally in this phase. Full source sizes are unknown until checked on EC2;
uncompressed tile estimates do not establish transfer sizes. Plan cumulative
raw/processed storage below 10 GB and total PC preview retrieval at or below
20 MB. Downloader per-call limits are not a cumulative storage budget.

The requested target remains 200–500 real pairs over at least three geographic
regions; 100 thoroughly reviewed pairs are preferable to forced counts. Accept
only high-confidence reviewed pairs in the first manifest. Retain medium/low
candidates in review, without training eligibility. Use 256×256 RGB tiles near
0.6 m only where source resolution supports that claim. Review pre/post tree
boundaries and actual absence/presence of new buildings; include reviewed
no-change and hard negatives. Static footprints and Hansen pixels are proposals,
not accepted temporal labels. Do not generate synthetic pairs.

Preserve source IDs, actual acquisition dates, bounds/CRS, original and output
resolution, dx/dy, class/confidence, reviewer/decision, all derived-source
licenses and `synthetic=false`. Keep whole source rasters/AOIs and all their
years together when geographically splitting. Existing loader uses separate
binary building/tree masks; a combined mask can accompany these. Audit actual
files, mask boundaries, duplicates, geographic leakage and registration before
reporting readiness. No training command, including a training dry run, may be
executed in this pilot phase.

## Requested acquisition report: actual state

| # | Field | Verified result |
| --- | --- | --- |
| 1 | Instance / region / SSH | Approved target missing; default-region tagged lookup empty. No SSH attempted. |
| 2 | Remote repository version | Not checked; local preparation starts from `7fc9f89`. |
| 3 | Remote environment | Not inspected or installed. |
| 4 | Remote downloaded bytes | 0 performed by this task; remote pre-existing data unknown. |
| 5 | EBS free space and dataset storage | Not measured; host access required. |
| 6 | Accepted real pairs | 0. Existing four local mock fixtures are not pilot data. |
| 7 | No-change / hard negatives | 0 real reviewed pairs. |
| 8 | New-building pairs | 0 real reviewed pairs; temporal construction evidence absent. |
| 9 | Tree-removal pairs | 0 real reviewed pairs. |
| 10 | Regions | 0 acquired/reviewed regions. Three-region target unmet. |
| 11 | Actual acquisition years / gaps | Not queried on EC2 or selected. Unit-test years are fixtures. |
| 12 | Resolution / bands / dimensions | Not measured on real pilot assets. |
| 13 | Alignment dx/dy and rejection | No real pilot measurements. |
| 14 | Confidence / rejected candidates | No candidates reviewed; no high/medium/low counts available. |
| 15 | Provenance / licenses | Candidate-source review above; no real per-pair approval. |
| 16 | Geographic split | No real train/validation split generated. |
| 17 | Duplicates / leakage / mask audit | No real pilot audit; earlier mock audit is not a substitute. |
| 18 | Readiness / future 25–100-step value | **NOT READY**; no evidence to justify training yet. |
| 19 | Preview / backup | No new real previews, EC2 retrieval or S3 backup. Bucket unspecified. |
| 20 | Prohibited operations / local transfers | No training, optimizer update, submission, instance/bucket creation/start, or large local data/weight/environment transfer. Small code and source-page inspection only. |

Verification: focused year-discovery/external/audit tests **74 passed**; Ruff
passed. Tests use controlled metadata fixtures and do not prove provider data
availability, reviewed real labels or remote execution.
