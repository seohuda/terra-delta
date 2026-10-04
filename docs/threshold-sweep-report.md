# Frozen baseline calibration on micro-pilot-v2

2026-10-04. Balanced improves the approximate local score from **0.470028 to 0.534947**, while no-change false positives fall from **12/17 to 6/17**. All three building and two tree-positive validation tiles retain presence TP. This is a calibration result on tiny AI-reviewed data, not a leaderboard estimate or evidence that learning is unnecessary.

Recommendation: **GET MORE DATA FIRST**. Keep balanced as the frozen-weight comparison for future training. Five positive tiles from one Washington area, residual no-change FP of 35.29%, and tree shape score 0.143536 cannot establish generalization. Add independently reviewed full-tile positives and difficult negatives across other regions before training. No optimizer was constructed, backward called, or training performed.

## Fixed inputs and behavior

- Experiment code Git SHA: `fec5d9a4f8f7914a20386b64c25c3c6f2c3e38e8`. Final reporting/config delivery SHA is reported with the delivery message; no force push.
- Validation manifest: `/data/terradelta/processed/micro-pilot-v2/val.csv`; 22 full tiles: building 3, tree 2, no-change 17. Seven partial training labels are outside this validation.
- Checkpoint: `/data/terradelta/baseline/unet_r18_cd.pt`; SHA-256 before and after: `1be127bd2f70c8e3b0efdc48c6d3effc29f30f090746a91c87e6c23e9b3e878b`.
- SMP UNet/ResNet18, six PRE/POST RGB channels, three classes, ImageNet normalization, CPU identity inference, no alignment. Organizer metadata steps=100 describes the existing weights.
- Official reference: `configs/inference_baseline.yaml`, argmax, min_area=30, min_pos_area=20, simplify=0.5, ndigits=2. Cached reproduction: `0.470028111067598`; delta from prior audit = 0.
- Each positive must pass its threshold, then competes with background and eligible other positives. Highest probability wins; background wins ties; building wins positive ties. Morphology conflicts use original positive probabilities. Argmax behavior is unchanged.
- Reference polygon filtering remains before simplification/exterior export. The cached component adapter passed canonical parity, with no optimized-backend substitution. Metrics use exported exteriors, including filled interior holes.

## Comparison

| Preset | Overall | Building | Tree | No-change FP |
| --- | ---: | ---: | ---: | ---: |
| official baseline | 0.470028 | 0.493284 | 0.446773 | 12/17 (70.59%) |
| conservative | 0.455355 | 0.435432 | 0.475277 | 4/17 (23.53%) |
| balanced | 0.534947 | 0.594617 | 0.475277 | 6/17 (35.29%) |
| aggressive | 0.534947 | 0.594617 | 0.475277 | 6/17 (35.29%) |

All three central presets retain building TP=3, tree TP=2, FN=0. Conservative meets the 30% FP target, but misses most building extent: building shape score falls from 0.588672 to 0.023643. It is an optional strict alarm filter, not the preferred segmentation reference. Balanced and aggressive have identical central metrics here; balanced lowers tree threshold to avoid the neighboring tree-recall cliff. Balanced misses the 30% FP target and still loses some building boundary coverage (shape 0.448058 versus 0.588672).

## Exactly three calibrated configurations

| Config | Building threshold | Tree threshold | Min area building/tree | Min positive area building/tree | Simplify px |
| --- | ---: | ---: | ---: | ---: | ---: |
| conservative | 0.80 | 0.35 | 20 / 100 | 20 / 20 | 0.25 |
| balanced | 0.55 | 0.30 | 100 / 100 | 20 / 20 | 0.25 |
| aggressive | 0.55 | 0.35 | 100 / 100 | 20 / 20 | 0.25 |

Files: `configs/inference_sweep_conservative.yaml`, `configs/inference_sweep_balanced.yaml`, `configs/inference_sweep_aggressive.yaml`. All explicitly use reference polygons, ndigits=2, identity TTA, no alignment; opening/closing/dilation/erosion=0, fill_holes=false. Numeric settings are canonical nested class settings, with no inert duplicate aliases. Equal-metric ties prefer original numerical defaults: min_pos_area=20; simplify=0.25 was closer to 0.5 than the equally scoring 0.0.

## Presence, boundaries and size

TP/FP/FN/TN are class-specific over all 22 tiles. Building's 13 baseline FP are over 19 class-negative tiles, not a denominator of 17. No-change FP uses only the 17 tiles with both GT classes absent. Predicted presence uses the evaluator's mandatory exported union area >=20 px², regardless of the configured pre-export minimum. Area/polygon means are over all 22 tiles, including empty predictions.

| Preset | Class | Macro presence F1 | Shape | TP/FP/FN/TN | Predicted positive tiles | Mean area px² | Mean polygons |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| official baseline | new_building | 0.397895 | 0.588672 | 3/13/0/6 | 16 | 4460.93 | 2.4091 |
| official baseline | tree_removal | 0.745174 | 0.148372 | 2/3/0/17 | 5 | 161.09 | 0.5909 |
| conservative | new_building | 0.847222 | 0.023643 | 3/2/0/17 | 5 | 411.64 | 0.5000 |
| conservative | tree_removal | 0.807018 | 0.143536 | 2/2/0/18 | 4 | 142.23 | 0.3182 |
| balanced | new_building | 0.741176 | 0.448058 | 3/4/0/15 | 7 | 3116.95 | 0.8182 |
| balanced | tree_removal | 0.807018 | 0.143536 | 2/2/0/18 | 4 | 142.23 | 0.3182 |
| aggressive | new_building | 0.741176 | 0.448058 | 3/4/0/15 | 7 | 3116.95 | 0.8182 |
| aggressive | tree_removal | 0.807018 | 0.143536 | 2/2/0/18 | 4 | 142.23 | 0.3182 |

## Search and stable selection

169 coarse threshold pairs: 0.30..0.90 by 0.05, no morphology. Three promising threshold regions received independent 9x9 min_area tests (10/20/30/40/50/75/100/150/200), producing 243 rows. Two retained regions received independent 7x7 min_pos_area tests (10/20/30/50/75/100/150), producing 98 rows. Six shared simplification tolerances (0/0.25/0.5/0.75/1/1.5) on two finalists produced 12 rows. This is staged selection, not an exhaustive/global optimum claim. Class area and total area are independent; simplification is shared to limit tiny-set tuning.

Threshold centers within ±0.05 of finalists were compared by neighboring presence recall and score spread. Balanced requires a recall-preserving stable neighborhood, prefers a center that does not regress the baseline, then lower central FP and higher central score. Conservative minimizes central FP subject to full presence recall; aggressive maximizes central score subject to full presence recall. 564 unique cached trials include baseline and plateau checks. Morphology was skipped because threshold/area already gave meaningful score and FP improvement.

Tree threshold 0.40 loses both positive tiles at these settings. Selecting the isolated 0.35 central value therefore gives a fragile aggressive preset. Balanced's 0.30 center keeps all nine neighbors at FN=0.

| Building threshold | Tree threshold | Overall | No-change FP | Building/tree FN |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.25 | 0.525843 | 8/17 (47.06%) | 0 / 0 |
| 0.50 | 0.30 | 0.525843 | 8/17 (47.06%) | 0 / 0 |
| 0.50 | 0.35 | 0.525843 | 8/17 (47.06%) | 0 / 0 |
| 0.55 | 0.25 | 0.534947 | 6/17 (35.29%) | 0 / 0 |
| 0.55 | 0.30 | 0.534947 | 6/17 (35.29%) | 0 / 0 |
| 0.55 | 0.35 | 0.534947 | 6/17 (35.29%) | 0 / 0 |
| 0.60 | 0.25 | 0.513598 | 6/17 (35.29%) | 0 / 0 |
| 0.60 | 0.30 | 0.513598 | 6/17 (35.29%) | 0 / 0 |
| 0.60 | 0.35 | 0.513598 | 6/17 (35.29%) | 0 / 0 |

Balanced score range **0.513598–0.534947**, all five positive presences retained. Stable under the recorded criterion (no neighbor FN; score spread <=0.05). FP ranges 35.29–47.06%: presence/score stability does not imply that the 30% FP target is met in neighbors.

## Sample changes and previews

Balanced: building FP fixed **9**, still **4**; tree FP fixed **1**, still **2**. Building TP preserved **3**, tree TP preserved **2**; TP lost/new FN/new FP **0** for both. These are class events; combined no-change false-positive tiles fall **12 → 6**.

`per_sample_changes.csv` on EBS contains all 44 class/sample rows and the coarse threshold pairs that remove each of the 13 baseline building FP. Six-column preview is PRE | POST | GT | baseline | balanced | conservative, with five positive and five hard-negative rows. Red=building, cyan=tree. Native comparison shows conservative fragments on broad new roofs, incomplete detection of removed tree areas, and persistent FP on unchanged roofs/nearby objects. Small comparison JPG was inspected; raw imagery, weights and probability maps were not relayed through the PC.

| ID | Class | Balanced transition | Baseline area | Balanced area |
| --- | --- | --- | ---: | ---: |
| wa_olympia_r01_c04 | new_building | FP_fixed | 231.00 | 0.00 |
| wa_olympia_r02_c10 | new_building | FP_still | 10598.00 | 8546.00 |
| wa_olympia_r03_c08 | new_building | FP_fixed | 96.00 | 0.00 |
| wa_olympia_r04_c04 | new_building | FP_fixed | 395.00 | 0.00 |
| wa_olympia_r04_c06 | new_building | FP_still | 4131.00 | 3199.00 |
| wa_olympia_r04_c08 | new_building | FP_fixed | 40.00 | 0.00 |
| wa_olympia_r05_c07 | new_building | FP_fixed | 249.00 | 0.00 |
| wa_olympia_r06_c07 | new_building | FP_fixed | 55.00 | 0.00 |
| wa_olympia_r07_c10 | new_building | FP_fixed | 94.50 | 0.00 |
| wa_olympia_r10_c08 | new_building | FP_still | 20273.00 | 16166.00 |
| wa_olympia_r14_c07 | new_building | FP_still | 1316.00 | 637.00 |
| wa_olympia_r11_c01 | new_building | TP_preserved | 21998.00 | 17194.00 |
| wa_olympia_r14_c00 | new_building | TP_preserved | 16860.00 | 10744.00 |
| wa_olympia_r14_c01 | new_building | TP_preserved | 21344.00 | 12087.00 |
| wa_olympia_r06_c02 | new_building | FP_fixed | 389.00 | 0.00 |
| wa_olympia_r12_c06 | new_building | FP_fixed | 71.00 | 0.00 |
| wa_olympia_r03_c08 | tree_removal | FP_still | 402.00 | 403.00 |
| wa_olympia_r05_c06 | tree_removal | FP_fixed | 96.00 | 0.00 |
| wa_olympia_r07_c10 | tree_removal | FP_still | 736.00 | 563.00 |
| wa_olympia_r06_c02 | tree_removal | TP_preserved | 2083.00 | 2006.00 |
| wa_olympia_r12_c06 | tree_removal | TP_preserved | 227.00 | 157.00 |

## Proof and retention

59 focused calibration/postprocessing/search tests passed. Full suite with optimizer-construction and backward guards: **448 passed**, 59 pre-existing rasterio/Affine deprecation warnings; `tests/test_training.py` excluded because some tests construct optimizers. Ruff over src/scripts/tests and git diff check passed. Tests cover exclusive ties/background/fallback/morphology conflicts, per-class area, empty masks, compressed caches, existing direct/cached forward equivalence, reference polygon reuse, stable selection, config export and byte-identical resume.

Verification: **36 files byte-identical** after an actual EC2 rerun, including CSV/YAML/JSON/journal, all 22 compressed maps and the comparison JPEG. Cache mtimes also stayed unchanged. All validation PRE/POST/mask SHA-256s and manifest hash stayed unchanged. Canonical reference conversion independently reproduced the baseline and all three preset metrics from cache.

Cache: 14,837,237 bytes, float32 compressed NPZ; preview: 861,537 bytes. Runtime: torch 2.14.1+cpu, NumPy 2.5.2, Shapely 2.1.2.

Optimizer constructions=0; optimizer steps=0; backward=0. GPU use=0. Competition submissions=0. Large new downloads=0. Only the existing `i-0766a472ecb5bcf88` CPU builder was started; no other project resource was modified.

EBS output root: `/data/terradelta/outputs/threshold-sweep-v1/`. It contains probability-cache, coarse_thresholds.csv, area_search.csv, positive_area_search.csv, final_search.csv, plateau_centers.csv, robustness.csv, best_configs.json, per_sample_changes.csv, three YAMLs, previews, journal/identity/logs and verification.json. Earlier interim selections remain in small archive subdirectories; all share the same 22 probability maps. Raw imagery, checkpoint and maps stay out of Git.

Encrypted 30 GB gp3 `vol-0070845086ec08190` is retained with DeleteOnTermination=false. End-of-run procedure: publish/pull small code/config/report, flush EBS, revoke the temporary read-only deploy key, stop the builder without termination, and verify `stopped` through the EC2 API. The final delivery message reports the observed EC2 state and final Git SHA.
