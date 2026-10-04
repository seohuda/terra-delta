# AIHub 71363 review — 2026-10-05

Dataset: **AI기반 국립공원 변화탐지 모니터링 플랫폼 구축**, AIHub dataset71363, version1.2 (2024-05-31).

## Permission evidence

The [organizer's September18 notice for topics3 and4](https://aifactory.space/en/competitions/9307/posts/notice/241) explicitly identifies dataset71363 as an optional training reference, subject to AIHub terms and licensing. It says SkySat0.5m resembles the topic3 evaluation resolution0.55–0.7m and points to drone0.1m for building/logging detail. This specific guidance does not grant independent Planet/SkySat licensing or permission to redistribute raw images.

The [competition rules](https://aifactory.space/ko/competitions/9306) allow external data, require commercial/derivative/redistribution rights, and transfer winning models, including weights and derived results. Training datasets are excluded from that transfer. The [AIHub policy](https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) permits commercial/noncommercial AI research and development, requires attribution to NIA, restricts originals/third-party access and foreign export, and requires identity/purpose/download approval. The [official FAQ](https://aihub.or.kr/aihubnews/faq/list.do?currMenu=146&topMenu=104) permits commercial sale/distribution of trained models and other learned results with the exact dataset name and AIHub attribution; original and merely edited data redistribution remains prohibited.

**Decision:** organizer-supported training reference under AIHub's approved access and learned-result terms. It is not an unrestricted CC0 source. Provider terms and each downloaded subset still govern; the original data cannot enter Git, a public report, or the submission ZIP. Keep any downloaded data on Seoul-region EC2. Record NIA/AIHub attribution in model notices. Do not infer independent commercial satellite rights from the portal.

## Useful and unsuitable subsets

The [provider's data description](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71363) describes static1024×1024 land-cover segmentation, TIF images/labels and JSON metadata; it does not establish reviewed PRE/POST change ground truth.

| Subset | Relevance | Required treatment |
| --- | --- | --- |
| SkySat0.5m | Buildings/forest, similar resolution to competition | Useful appearance source; confirm bands, georeferencing, metadata, exact label IDs and rights. Static building does not mean new building. |
| Drone0.1m | Building and logging-area labels | Useful object/texture source after resampling to target physical resolution. Static logging area does not mean newly removed canopy. |
| Sentinel-2 10m | Forest/logging/land cover | Too coarse to serve as target-resolution building shape GT; not a replacement for high-resolution verifier examples. |
| Landsat30m | Broad land-cover labels | Exclude from high-resolution change-positive supervision. |
| Forest/road/farm/water/shadow/static building | Hard-negative appearance candidates | A true no-change pair requires identical-scene synthesis with declared augmentation, or two dates with full review. Static semantic absence alone is insufficient. |

Any eventual change examples must either be real aligned temporal pairs with reviewed directional labels, or explicitly marked synthetic edits with known changed regions. Preserve parent image/geographic groups when splitting; exclude local WA validation overlap. Reuse the existing synthetic generator for compatible sources rather than relabeling static masks as temporal changes.

## Observed access and bounded acquisition

The user logged into AIHub on October5. Browser login does not authenticate the separate EC2 download. The official API guide requires an AIHub API key and dataset-specific approval. The user has been asked to issue the key and save it locally with mode600; that credential is still pending. An unauthenticated, bounded EC2 sample request returned a login page, not an image archive. No AIHub data is present on retained EBS. The public policy says manuals do not require login, but the two direct manual links returned a79-byte access-error page in this session. No PDF was obtained. Public description/policy/FAQ/organizer HTML were preserved and hashed on EBS; see `v22-aihub-evidence.json`. No original image download or AIHub training use is claimed.

The official AIHub shell was inspected on EC2 (version0.6, SHA256 `3475a89b89ca10cdebfd7ef0542ec54650759bd5c15491e4dc0da6c15d93390e`). It uses the API-key header and selected `fileSn` download endpoint. Prepared acquisition restricts dataset71363 and explicit file keys, bounds payload bytes/free space, and keeps provider payloads entirely on Seoul EC2. The live public file list reports98.68GB total. The retained30GiB disk has about17GiB free, so downloading all data or the full42.91GB SkySat training ZIP is unsuitable. If approved access becomes available, first inspect a bounded sample/metadata, then acquire only a fitting subset. Relevant listed files:

| File | Key | Reported size |
| --- | --- | --- |
| SkySat metadata, Other | 533615 | 10.24MB |
| Drone metadata, Other | 533614 | 8.33MB |
| SkySat source, Training | 491162 | 42.91GB — exceeds disk capacity |
| SkySat source, Validation | 491174 | 5.54GB |
| SkySat label, Validation | 491178 | 16.47MB |
| SkySat JSON, Validation | 491182 | 110.26MB |
| Drone source, Validation | 491173 | 4.17GB |
| Drone label, Validation | 491177 | 37.51MB |
| Drone JSON, Validation | 491181 | 1.77MB |

The provider's Training/Validation labels describe its own release. They do not authorize mixing our selection data into verifier training. Create a source-group split before any auxiliary fitting; do not call a provider validation image an independently tested change example.
