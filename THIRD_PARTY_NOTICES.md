# Third-Party Software and Data Notices

This repository incorporates code and references assets subject to the third-party licenses and notices described below.

---

## 1. SatlasPretrain Models (Allen Institute for AI)

Portions of `src/terradelta/models/satlas_v3.py` (specifically checkpoint state-dict adjustment utilities) are adapted from `allenai/satlaspretrain_models/utils.py` at commit `7b5cd45adc3cad70b3834d65956974af6f6bffd0`.

- **Copyright:** (c) 2023 Allen Institute for AI
- **License:** Apache License, Version 2.0
- **License text:** A full copy of the Apache-2.0 license is available in [`third_party/satlaspretrain_models.LICENSE`](third_party/satlaspretrain_models.LICENSE).
- **Modifications:** Added type annotations, condensed docstring, explicit guard for prefix counts, and integration into the TerraDelta Siamese inference pipeline.

---

## 2. Competition Baseline (AIFactory / Korea National Park Service)

Reference utilities for polygon rasterization and competition IoU metric evaluation in `src/terradelta/postprocess/` and `src/terradelta/metrics/` follow the contract defined by the baseline notebook for the **2026 National Park Satellite Monitoring AI Challenge (Topic 3: Illegal Building & Tree Removal Detection)**.

- **Source:** [AIFactory Competition 9306](https://aifactory.space/ko/competitions/9306)
- **Notice:** Baseline code is referenced solely for submission format parity and evaluation compliance. Raw baseline weights and proprietary competition test sets are excluded from this repository.

---

## 3. Dataset Attribution & Non-Redistribution Notice

- **AIHub Dataset 71363:** Auxiliary multi-temporal satellite imagery was acquired under legitimate domestic Korean access terms from the National Information Society Agency (NIA) AIHub platform.
- **Redistribution Policy:** In compliance with dataset terms and challenge guidelines, **no raw satellite imagery, segmentation masks, intermediate crop files, trained model weights (`.pt`/`.pth`), or competition submission archives (`.zip`) are redistributed or hosted in this repository.** Only source code, inference configurations, and evaluation reports are made publicly available under the MIT License.
