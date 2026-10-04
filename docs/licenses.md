# Permission evidence and API audit

Reviewed 2026-10-04. [LICENSE_DATA.md](../LICENSE_DATA.md) records commercial use,
derivatives, redistribution, attribution and actual usage. This inventory does
not grant new rights or approve data solely because an endpoint is public.

The dry-run proof below is the initial PC verification. Subsequent acquisition
is documented in the [actual CPU pilot report](real-pilot-report.md): six NAIP
rasters, two Hansen lossyear rasters and three Microsoft GlobalML gzips were
downloaded directly on EC2. Provider pages and CDLA-Permissive-2.0 were retained.
NAIP supplies 644 candidates and 71 approved no-change pairs. Hansen and
Microsoft supply mining/reference evidence only, with zero final GT masks.
FEMA remains excluded. No training occurred; positive mask quality and model
quality remain unverified.

## Organizer baseline and encoder weights

Preserved original LICENSE grants baseline use, modification, retraining and
redistribution for competition participation; other uses follow competition rules.
NOTICE identifies the embedded ResNet18 ImageNet encoder from torchvision/SMP.
Organizer materials used for fine-tuning are not bundled. Original files and
checkpoint remain unchanged locally, with SHA-256 recorded in baseline/integrity.json.

Bundled dependency notices identify torchvision BSD-3-Clause, SMP MIT, Shapely
BSD-3-Clause and dynamically linked GEOS LGPL-2.1. Primary software licenses:
[torchvision](https://github.com/pytorch/vision/blob/v0.29.0/LICENSE),
[SMP 0.5.0](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/LICENSE).
Retain original notices in exported baseline-derived packages.

The standalone [SMP encoder card](https://huggingface.co/smp-hub/resnet18.imagenet)
labels its license `other`. [Raw ImageNet terms](https://image-net.org/download-images)
restrict supplied image access to noncommercial research/education. Software
licensing alone does not resolve upstream image rights. No new encoder or ImageNet
image download occurred; inference uses encoder_weights=None and the organizer's
local competition checkpoint. Standalone weights need separate review.

## External permission evidence

| Source | Primary evidence | Pipeline decision |
| --- | --- | --- |
| NAIP | [USDA catalog license field](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery), [USGS product](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip) | Named NAIP public-domain evidence; retain exact asset provenance and review jurisdiction |
| FEMA USA Structures | [Service item JSON](https://www.arcgis.com/sharing/rest/content/items/0ec8512ad21e4bb987d7e848d14e7e24?f=json), licenseInfo/access/accessInformation | Text and legal-code links say CC BY 4.0; badge URL says BY 3.0. Discovery reports unknown; resolve exact attribution/version terms before approving samples |
| Hansen v1.12 | [Versioned provider license/attribution](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Commercial derivatives/redistribution with credit, license link and change disclosure; candidate mining only |
| LEVIR-CD | [Provider Usage section](https://justchenhao.github.io/LEVIR/) | Academic only, commercial prohibition; excluded |
| AIHub | [Official portal](https://www.aihub.or.kr/); no selected dataset terms reviewed | Unresolved, excluded by default |

FEMA's complete item metadata read was capped at 65,536 bytes, returned HTTP 200
with 9,507 body bytes, and recorded only permission/access fields. It credits ORNL
and FEMA Geospatial Response Office. Public access and an empty layer copyright
field do not make the item public domain. No badge image or geometry was read.

[Competition rules](https://aifactory.space/ko/competitions/9306) apply external
permission requirements to pretrained weights and inherited training-data
restrictions as well as data. Supplied competition data is for participation;
never publish it automatically. [USAGov guidance](https://www.usa.gov/government-copyright)
distinguishes federal works from third-party/state materials and international
rights. The public-domain NAIP entry is not a blanket grant for every government
basemap or any imagery served by Microsoft.

## Live metadata and dry-run proof

Production CLIs succeeded using a small Maryland AOI
[-76.61,39.29,-76.60,39.30], NAIP year 2021. They use a 4,000,000-byte per-response
metadata cap; no imagery, thumbnails, previews, or footprint geometry were fetched.
Raster sizes come from HEAD only. SAS URLs were used transiently and are absent
from persisted output reports.

| Local report (ignored by Git) | Verified result | Limit |
| --- | --- | --- |
| outputs/naip-dry-run.json | Item md_m_3907644_sw_18_060_20210617, acquisition 2021-06-17, 0.6 m, EPSG:26918, shape 12,390×9,860×4; signed HEAD 447,829,958 bytes | No TIFF GET; whole COG exceeds default 50 MB file/100 MB total download caps |
| outputs/fema-dry-run.json | 218 AOI features; record cap 2,000; OBJECTID; advertised pagination true; 327,000-byte geometry estimate; license_status unknown | No geometry/pages fetched; estimate is heuristic, not wire size |
| outputs/hansen-dry-run.json | GFC-2024-v1.12 lossyear 40N 080W; HEAD 42,217,473 bytes; raw estimate 1,296,000,000 | No asset GET or direct segmentation truth |

Endpoints and contracts:
[NAIP item](https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/md_m_3907644_sw_18_060_20210617),
[public STAC](https://planetarycomputer.microsoft.com/docs/),
[transient SAS signing](https://planetarycomputer.microsoft.com/docs/concepts/sas/),
[FEMA layer](https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0),
[Esri count/GeoJSON/offset query](https://developers.arcgis.com/rest/services-reference/enterprise/query-feature-service-layer/),
[Hansen granule](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_lossyear_40N_080W.tif).

During that initial verification, real FEMA pagination, downloaded raster
readability/tiling, high-resolution labels and model quality were unverified.
The later EC2 pilot proves actual raster decoding/tiling; FEMA geometry and
positive GT/model quality remain unverified. Offline mocks/local fixtures
cover the API and CRS contracts. A successful metadata plan is not acquisition,
permission approval, or a training outcome. Pinned Hansen v1.12 is superseded by
v1.13 according to the [provider catalog](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12);
upgrades need an explicit version/provenance review.

Before changing unknown to an approved manifest status, retain evidence for the
exact product and all constituent sources. Unknown terms and unreviewed labels
cannot enter default training. See [data pipeline](data-pipeline.md) for commands,
limits, scope masks and no-leak split behavior.
