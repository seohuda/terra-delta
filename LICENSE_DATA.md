# Data and pretrained weight permissions

Reviewed 2026-10-04. This is a provenance inventory, not a new license for third-party
data or weights. Initial checks were metadata-only. The subsequent CPU pilot
downloaded six NAIP rasters, two Hansen lossyear rasters and three Microsoft
GlobalML footprint gzips directly on EC2. Its 71 approved pairs use NAIP and
reviewed absence masks; no positive GT was approved. No training or optimizer
steps were executed. [Actual usage report](docs/real-pilot-report.md).

The [competition's external-data rules](https://aifactory.space/ko/competitions/9306)
require commercial use, derivatives and redistribution rights for external data
and pretrained weights, including restrictions inherited from their training data.
Winning models are subject to transfer; training datasets themselves are excluded
from that transfer. Access without payment does not establish these rights.

## Initial preparation inventory (before EC2 acquisition)

| Dataset/model | Provider | Official source | License | Commercial use | Derivative works | Redistribution | Attribution | Actual usage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAIP imagery | USDA FPAC/USGS; Microsoft hosts STAC/COGs | [USDA catalog](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) | Catalog explicitly links public-domain designation | Supported by that designation; jurisdiction and individual asset review remain | Supported on same basis | Supported on same basis | Retain USDA/USGS, item ID, acquisition date and metadata | Maryland 2021 STAC, transient signing and HEAD; no raster GET | Applies to NAIP assets, not every government-hosted or commercial basemap layer |
| USA Structures public layer | FEMA/DHS, ORNL, USGS | [Service item terms](https://www.arcgis.com/sharing/rest/content/items/0ec8512ad21e4bb987d7e848d14e7e24?f=json), [DHS/FEMA catalog](https://catalog.data.gov/dataset/usa-structures) | Item text and deed/legal-code links: CC BY 4.0; catalog points to government-works guidance | Yes under stated CC BY 4.0 terms | Yes under those terms | Yes under those terms | ORNL; FEMA Geospatial Response Office; preserve map/source attribution | Metadata and AOI count only; no footprints | Badge URL says BY 3.0 while text/links say 4.0; preserve evidence and resolve inconsistency before training/release; discovery stays unknown and is excluded until reviewed |
| Hansen GFC 2000–2024 v1.12 | Hansen/UMD/Google/USGS/NASA | [Versioned provider page](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html) | CC BY 4.0 | Yes, under license | Yes, under license | Yes, under license | Credit, license link, change notice; see below | URL planning, mocked candidate grids and HEAD only | Candidate mining only; no direct high-resolution ground truth; superseded version remains pinned |
| Official UNet ResNet18 checkpoint and notebook | AIFactory | [Baseline Drive](https://drive.google.com/file/d/1RZjoG0hITfY5XqIPuv5KtKYV01XUiOT3/view); local original LICENSE/NOTICE | Competition-specific permission plus bundled dependency notices | Outside competition: unresolved; follow rules | Competition use/modification/retraining explicitly allowed | Competition redistribution allowed by bundled LICENSE; retain notices | Preserve original LICENSE and NOTICE | Preserved local reference; license review read original notices | Permission for original package does not authorize public release of competition imagery |
| ResNet18 ImageNet encoder parameters embedded in official baseline | torchvision; SMP redistribution | Local baseline NOTICE; [torchvision v0.29.0 BSD license](https://github.com/pytorch/vision/blob/v0.29.0/LICENSE), [SMP weight card](https://huggingface.co/smp-hub/resnet18.imagenet) | Baseline attributes encoder to BSD-3-Clause; standalone card labels license `other` | Bundled baseline competition scope supported; standalone commercial clearance unresolved | Same qualification | Keep BSD notices; resolve standalone weight/data terms first | torchvision copyright/license; record exact weight source | Embedded in existing checkpoint, no new weight fetch | Software license alone does not settle upstream image rights; `encoder_weights=None` remains default |
| ImageNet raw images | ImageNet and individual image rightsholders | [Official download terms](https://image-net.org/download-images) | Noncommercial research/education access; underlying images owned by others | Not approved | Not approved for this pipeline | Not approved | Dataset/provider terms would apply | Not downloaded or included | Distinct from the official competition checkpoint permission |
| LEVIR-CD; LEVIR-CD+ family | LEVIR/Beihang | [Provider's Usage section](https://justchenhao.github.io/LEVIR/) | LEVIR-CD: academic only, commercial use prohibited | No for LEVIR-CD | Outside permitted academic scope: no | No default approval | Academic citation is not commercial permission | Excluded; no download | Family variants remain blocked until separate compatible terms are demonstrated |
| AIHub datasets | NIA and dataset-specific providers | [Official AIHub portal](https://www.aihub.or.kr/) | Dataset-specific; not verified for a selected dataset | Unresolved | Unresolved | Unresolved | Dataset-specific | Excluded; no download | Portal availability is not proof of competition or redistribution compatibility |
| Local test fixtures and synthetic/mock arrays | TerraDelta test authors | `tests/test_dataset.py`, `tests/test_external.py`, `tests/test_synthetic.py` | Generated locally; no external imagery | Subject to repository code terms | Same | Same | Describe synthesis and inherited inputs if real data are later used | Tiny temporary PNG/GeoTIFF fixtures and arrays | Synthetic edits of real NAIP/FEMA retain input provenance and obligations |

## Initial PC dataset audit (2026-10-04, before EC2 acquisition)

The only persistent sample imagery on the PC is four `synthetic_mock` pairs in
`outputs/smoke/input/images/`. `scripts/smoke_test.py` creates them from seeded
NumPy random RGB arrays, two constant-color rectangular edits and binary masks.
They contain no NAIP, FEMA, Hansen, LEVIR-CD or AIHub imagery/geometry. Their
2020/2022 dates and `mock_region_*` locations are invented fixture metadata.
The manifest's `commercial_ok`/`reviewed` assertions do not establish a real
training dataset. Audit records mark all four unverified and training-ineligible.

| Actually used item | Evidence | Rights/review status | Training usage |
| --- | --- | --- | --- |
| Generated RGB arrays and masks | `scripts/smoke_test.py`, `outputs/smoke/manifest.csv` | Local author-generated fixtures; repository terms apply; no real-world provenance | Diagnostic only, all four excluded |
| Organizer checkpoint | Unchanged local baseline LICENSE/NOTICE and integrity hashes | Existing competition-specific grant; retain bundled notices; no new weight download | CPU inference only; zero optimizer updates |
| NAIP / FEMA / Hansen | Existing metadata-only plan JSONs | Planned sources, not used sample inputs; prior inventory remains a plan | No acquired imagery/footprints and no training use |

That table describes the initial PC audit. The subsequent EC2 source inventory
is below; it does not change the mock audit's facts. Future
synthetic samples inherit the licenses of their real imagery/footprint inputs;
they cannot inherit the mock fixture row's status. [Audit report](docs/data-audit.md).

## Actual EC2 source usage

| Source | Official evidence | License / permissions | Actual usage |
| --- | --- | --- | --- |
| NAIP | [USDA catalog](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) | Public-domain designation supports use, derivatives and redistribution on that basis; retain exact USDA/product/date provenance | Six raster downloads; 644 temporal candidate pairs; 71 approved no-change pairs; no external positive label mask incorporated |
| Hansen GFC v1.12 | [Versioned provider](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Commercial use, derivatives and redistribution with credit, license link and change disclosure | Two lossyear rasters; candidate mining only; zero final GT pairs |
| Microsoft GlobalML 2026-07-24 | [GlobalML license](https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/LICENSE), [agreement](https://cdla.dev/permissive-2-0/) | CDLA-Permissive-2.0 permits use and derivatives; include agreement when sharing data; retain Microsoft/product/source provenance | Three quadkey gzips streamed for static references; zero final GT pairs; distinct from USBuildingFootprints ODbL product |
| FEMA | Item text/badge inconsistency documented above | Remains excluded until exact version/attribution terms are resolved | Zero geometry downloads or pilot pairs |

Provider license pages and the CDLA agreement are preserved in
`/data/terradelta/metadata` on the retained EBS. Final zero masks encode reviewed
absence facts from NAIP; no Hansen or Microsoft mask was promoted to GT.
Per-pair metadata identifies **Codex AI visual review**, not independent human
annotation. No model fitting used any of these sources.

## Hansen attribution (used for EC2 candidate mining)

When displaying data use `Source: Hansen/UMD/Google/USGS/NASA`. Link the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) and the
[provider's preferred dataset landing page](https://glad.earthengine.app/view/global-forest-change).
For derived products record version `GFC-2024-v1.12` and describe candidate
selection, reprojection and any other changes. The [versioned provider page](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html)
supplies the required credit and citation; cite Hansen et al. (2013),
*High-Resolution Global Maps of 21st-Century Forest Cover Change*, Science
342:850–853, [doi:10.1126/science.1244693](https://doi.org/10.1126/science.1244693).

## Gate and remaining review

`terradelta.data.pairing.training_eligibility` accepts only explicit
`public_domain`, `cc_by_4.0`, `commercial_ok` or `verified_commercial` statuses
with nonempty source provenance. It blocks LEVIR/AIHub, direct Hansen labels,
and `candidate`, `weak_unreviewed` or `unreviewed` label statuses. These strings
are assertions made by a reviewer; the code does not verify a legal document.
Missing label-review metadata is not itself rejected by this gate. Record the
actual review decision and reviewer rather than relying on that permissive case.

For mixed inputs, a single row status must reflect review of **all** inputs.
Keep source URLs, versions, retrieval dates, license evidence, acquisition dates,
review scope and synthetic parent IDs with local manifests. Unknown terms remain
excluded; downloader availability does not make a sample training eligible.

The [US government guidance](https://www.usa.gov/government-copyright) distinguishes
federal employee works from third-party and state/local material, and notes that
international rights can differ. The public-domain NAIP entry records evidence
for that product, not a blanket worldwide clearance. Before actual
training/transfer, confirm applicable asset terms and competition compatibility.
See [the detailed audit](docs/licenses.md) and [pipeline contract](docs/data-pipeline.md).
