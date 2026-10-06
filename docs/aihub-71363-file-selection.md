# AIHub Dataset 71363: File Selection & Fast-Path Inventory

**Dataset**: 71363 (`310.AI기반 국립공원 변화탐지 모니터링 플랫폼 구축`)  
**Task Target**: Temporal PRE/POST Change Detection (GSD ~0.55–0.70m/px, `new_building`, `tree_removal`)  
**Network**: Korean Domestic School Network (`211.169.190.150`, AS3786 LG DACOM Corporation)  
**Strategy**: Deadline Fast-Path (Reserve >= 45m for final packaging/submission; download budget < 90m)

---

## 1. Full Dataset 71363 Inventory & Semantic Classification

| Filekey | Archive / File Name | Size | Modality / Sensor | Annotation Type | Relevance to new_building | Relevance to tree_removal | Semantic Trust Level | Fast-Path Selection |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **491162** | `TS_02. Skyset.zip` | 43 GB | Satellite (SkySat, 0.5m) | Raw RGB Imagery | High (0.5m resolution) | High (0.5m resolution) | DIRECT_IMAGERY | EXCLUDE (Exceeds 29GB disk & 90m budget) |
| **491161** | `TS_01.Drone.zip` | 39 GB | Aerial (Drone, 0.1m) | Raw RGB Imagery | Medium (0.1m resolution) | Medium (0.1m resolution) | DIRECT_IMAGERY | EXCLUDE (Exceeds 29GB disk & 90m budget) |
| **491163** | `TS_03. Sentinel2.zip` | 1 GB | Satellite (Sentinel-2, 10m) | Raw Multispectral | None (10m GSD too coarse) | Low (coarse forest index) | AUXILIARY_ONLY | EXCLUDE (Coarse GSD) |
| **491164** | `TS_04. Landsat.zip` | 401 MB | Satellite (Landsat-8, 30m) | Raw Multispectral | None (30m GSD too coarse) | Low (coarse NDVI) | AUXILIARY_ONLY | EXCLUDE (Coarse GSD) |
| **491165** | `TL_01.LABEL_01.Drone.zip` | 612 MB | Aerial (Drone, 0.1m) | Raster TIF Masks | Medium | Medium | DERIVABLE_GT | OPTIONAL_TRAIN |
| **491166** | `TL_01.LABEL_02. Skyset.zip` | 94 MB | Satellite (SkySat, 0.5m) | Raster TIF Masks | High | High | DERIVABLE_GT | OPTIONAL_TRAIN |
| **491167** | `TL_01.LABEL_03. Sentinel2.zip` | 15 MB | Satellite (Sentinel-2) | Raster TIF Masks | None | None | AUXILIARY_ONLY | EXCLUDE |
| **491168** | `TL_01.LABEL_04. Landsat.zip` | 5 MB | Satellite (Landsat-8) | Raster TIF Masks | None | None | AUXILIARY_ONLY | EXCLUDE |
| **491169** | `TL_02.JSON_01.Drone.zip` | 23 MB | Aerial (Drone, 0.1m) | GeoJSON Vector | Medium | Medium | DERIVABLE_GT | OPTIONAL_TRAIN |
| **491170** | `TL_02.JSON_02. Skyset.zip` | 545 MB | Satellite (SkySat, 0.5m) | GeoJSON Vector | High | High | DERIVABLE_GT | OPTIONAL_TRAIN |
| **491171** | `TL_02.JSON_03. Sentinel2.zip` | 724 MB | Satellite (Sentinel-2) | GeoJSON Vector | None | None | AUXILIARY_ONLY | EXCLUDE |
| **491172** | `TL_02.JSON_04. Landsat.zip` | 667 MB | Satellite (Landsat-8) | GeoJSON Vector | None | None | AUXILIARY_ONLY | EXCLUDE |
| **491173** | `VS_01.Drone.zip` | 4 GB | Aerial (Drone, 0.1m) | Raw RGB Imagery | High (paired with VL_01) | High (paired with VL_01) | DIRECT_IMAGERY | **SELECTED (Fast-Path Tier 2)** |
| **491174** | `VS_02. Skyset.zip` | 6 GB | Satellite (SkySat, 0.5m) | Raw RGB Imagery | Highest (Exact 0.5m GSD) | Highest (Exact 0.5m GSD) | DIRECT_IMAGERY | **SELECTED (Fast-Path Tier 1)** |
| **491175** | `VS_03. Sentinel2.zip` | 151 MB | Satellite (Sentinel-2, 10m) | Raw Multispectral | None | Low | AUXILIARY_ONLY | EXCLUDE |
| **491176** | `VS_04. Landsat.zip` | 72 MB | Satellite (Landsat-8, 30m) | Raw Multispectral | None | Low | AUXILIARY_ONLY | EXCLUDE |
| **491177** | `VL_01.LABEL_01. Drone.zip` | 38 MB | Aerial (Drone, 0.1m) | Raster TIF Masks | High (Multi-temporal) | High (벌목지 90, 산림 80) | DERIVABLE_GT | **SELECTED (Probe Verified)** |
| **491178** | `VL_01.LABEL_02. Skyset.zip` | 16 MB | Satellite (SkySat, 0.5m) | Raster TIF Masks | Highest (건물 10) | Highest (산림 80) | DERIVABLE_GT | **SELECTED (Probe Verified)** |
| **491179** | `VL_01.LABEL_03. Sentinel2.zip` | 2 MB | Satellite (Sentinel-2) | Raster TIF Masks | None | None | AUXILIARY_ONLY | EXCLUDE |
| **491180** | `VL_01.LABEL_04. Landsat.zip` | 620 KB | Satellite (Landsat-8) | Raster TIF Masks | None | None | AUXILIARY_ONLY | EXCLUDE |
| **491181** | `VL_02.JSON_01. Drone.zip` | 2 MB | Aerial (Drone, 0.1m) | GeoJSON Vector | High (Multi-temporal) | High (벌목지 90) | DERIVABLE_GT | **SELECTED (Probe Verified)** |
| **491182** | `VL_02.JSON_02. Skyset.zip` | 110 MB | Satellite (SkySat, 0.5m) | GeoJSON Vector | Highest | Highest | DERIVABLE_GT | **SELECTED (Fast-Path Tier 1)** |
| **491183** | `VL_02.JSON_03. Sentinel2.zip` | 82 MB | Satellite (Sentinel-2) | GeoJSON Vector | None | None | AUXILIARY_ONLY | EXCLUDE |
| **491184** | `VL_02.JSON_04. Landsat.zip` | 58 MB | Satellite (Landsat-8) | GeoJSON Vector | None | None | AUXILIARY_ONLY | EXCLUDE |
| **533614** | `01.메타데이터_01.Drone.zip` | 8 MB | Metadata | JSON Catalog | High (Dates, coords, GSD) | High | AUXILIARY_ONLY | **SELECTED (Probe Verified)** |
| **533615** | `01.메타데이터_02. Skyset.zip` | 10 MB | Metadata | JSON Catalog | Highest (Dates, GSD 0.5m) | Highest | AUXILIARY_ONLY | **SELECTED (Probe Verified)** |
| **533616** | `01.메타데이터_03. Sentinel2.zip` | 1 MB | Metadata | JSON Catalog | None | None | AUXILIARY_ONLY | EXCLUDE |
| **533617** | `01.메타데이터_04. Landsat.zip` | 829 KB | Metadata | JSON Catalog | None | None | AUXILIARY_ONLY | EXCLUDE |
| **533618** | `02.SHP_01.Drone.zip` | 36 MB | Shapefile | Vector GIS | Medium | Medium | DERIVABLE_GT | EXCLUDE (Redundant with GeoJSON) |
| **533619** | `02.SHP_02. Skyset.zip` | 726 MB | Shapefile | Vector GIS | High | High | DERIVABLE_GT | EXCLUDE (Redundant with GeoJSON) |
| **533620** | `02.SHP_03. Sentinel2.zip` | 794 MB | Shapefile | Vector GIS | None | None | AUXILIARY_ONLY | EXCLUDE |
| **533621** | `02.SHP_04. Landsat.zip` | 675 MB | Shapefile | Vector GIS | None | None | AUXILIARY_ONLY | EXCLUDE |

---

## 2. Selected Minimum Useful Fast-Path Subset

To respect the **competition 5-hour deadline**, the **local disk budget (29 GB)**, and the **90-minute download limit**:

### Selected Packages:
1. **Tier 1 (Core SkySat 0.5m Target Domain)**:
   - `491174`: `VS_02. Skyset.zip` (6 GB raw images, SkySat 0.5m/px)
   - `491178`: `VL_01.LABEL_02. Skyset.zip` (16 MB raster masks, **already verified in probe**)
   - `491182`: `VL_02.JSON_02. Skyset.zip` (110 MB GeoJSON vectors)
   - `533615`: `01.메타데이터_02. Skyset.zip` (10 MB catalog, **already verified in probe**)
2. **Tier 2 (High-Resolution Drone Complement)**:
   - `491173`: `VS_01.Drone.zip` (4 GB raw images, Drone 0.1m/px)
   - `491177`: `VL_01.LABEL_01. Drone.zip` (38 MB raster masks, **already verified in probe**)
   - `491181`: `VL_02.JSON_01. Drone.zip` (2 MB GeoJSON vectors, **already verified in probe**)
   - `533614`: `01.메타데이터_01.Drone.zip` (8 MB catalog, **already verified in probe**)

### Time and Size Budget Summary:
- **Total Download Size**: ~10.18 GB
- **Observed Bandwidth**: ~10–15 MB/s
- **Estimated Download Time**: **~12–15 minutes** (well under 90-minute limit)
- **Local Disk Usage**: ~10.2 GB (local available: 29 GB, 100% safe)
- **Time Remaining for Audit + Training**: **> 4.5 hours** (preserves required 45m submission reserve)
