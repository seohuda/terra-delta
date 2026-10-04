# TerraDelta

**Satellite Change Detection for National Park Monitoring**

TerraDelta detects newly constructed buildings and tree removal in paired
high-resolution pre/post RGB imagery and exports pixel-coordinate polygons.
It prepares an end-to-end workflow for the
[2026 National Park Satellite Monitoring AI Challenge, topic 3](https://aifactory.space/ko/competitions/9306):
licensed data discovery → aligned temporal tiles → geographic validation → short
fine-tuning → probability inference → polygon CSV → offline submission ZIP.

**Current status:** code and CPU verification only. No model training, optimizer
updates, cloud GPU operations, large dataset downloads, or competition/debug
submissions have been performed. No competition score is claimed.

## Official baseline and model

Pre-training audit (2026-10-04): **NOT READY**. The local filesystem contains
four generated mock pairs and zero real training pairs. The earlier 1,000-pair
figure is a storage estimate. [Audit evidence and commands](docs/data-audit.md)
separate mock diagnostics from real validation. Large data/environment operations
belong on EC2; see [the hotspot-safe workflow](docs/aws-training.md).

The [official baseline archive](https://drive.google.com/file/d/1RZjoG0hITfY5XqIPuv5KtKYV01XUiOT3/view)
is preserved unchanged under `baseline/original/` locally. Its weights and files
are excluded from Git; retrieval and SHA-256 inventory are in
[baseline/README.md](baseline/README.md) and
[baseline/integrity.json](baseline/integrity.json).
[Baseline analysis](docs/baseline-analysis.md) records actual notebook, checkpoint,
licenses and I/O rather than assumptions.

```text
pre RGB  ── ImageNet normalization ──┐
                                    ├── 6 channels ── UNet / ResNet18 ── 3 logits
post RGB ── ImageNet normalization ──┘
                                      background / new_building / tree_removal
```

SMP 0.5.0 UNet matches the original state_dict keys. Inputs are 256×256 RGB;
pre then post are normalized separately with mean [0.485,0.456,0.406] and
std [0.229,0.224,0.225]. Inference initializes with `encoder_weights=None`, so
bundled checkpoints load without fetching any pretrained files. Both plain
state_dict and dictionaries containing state_dict are supported with strict
architecture/class metadata validation.

The actual organizer checkpoint contains `steps=100`, `seed=0`. This motivates
short **optimizer-step comparisons**, not a claim that 100 steps is optimal.
Baseline inference uses softmax argmax with identical polygon parameters for
both classes. TerraDelta also supports independent probability thresholds,
optional horizontal/vertical TTA, raw probability maps and bounded phase/ECC
registration. Baseline preset disables TTA/alignment and retains exact reference
polygon serialization. Conservative thresholds are untuned examples.

## Quick start — no training

Python 3.10+; create an isolated environment from the repository root. Install
an appropriate CPU or CUDA PyTorch wheel for your future host before dependencies
if you need to control wheel size. The preparation CPU tests use the local
`.venv`; no CUDA execution is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
# After retrieving the unchanged official baseline locally:
python scripts/inspect_baseline.py --forward
python scripts/smoke_test.py --output outputs/smoke_new --execute-notebook
```

The smoke command creates four tiny mock pairs, runs loader/forward/loss checks,
validation/search and local export, and optionally executes the generated
notebook in a separate CPU kernel. It never constructs a training optimizer or
performs optimizer updates. Mock scores only demonstrate code execution.
Checkpoint-dependent tests skip when the local official archive is absent;
ordinary Git checkouts do not carry model weights.

## Data and license policy

| Source | Role | Default policy |
| --- | --- | --- |
| NAIP | Real 0.3–1 m aerial imagery; AOI/year discovery, common-grid temporal RGB/NIR tiling | Review asset provenance and license; bounded dry-run first |
| FEMA USA Structures | Building footprints, CRS-aware rasterization, synthetic deletion references | Static footprint is not a temporal change label |
| Hansen Global Forest Change | 30 m loss event candidate mining | Never direct high-resolution segmentation GT; refine/review using NAIP |
| LEVIR-CD / AIHub | Potentially incompatible or unverified terms | Excluded from default training |

[LICENSE_DATA.md](LICENSE_DATA.md) records commercial use, derivatives,
redistribution and attribution separately for datasets/weights. Free access or
an open-source model library does not establish data/weight rights. Unknown
license status and unreviewed candidates are rejected before training reads
images. Official competition assets have their own restricted-purpose grant.

Downloaders default to metadata-only operation and expose `--dry-run`.
Payload transfer requires explicit `--download` with file/feature/byte limits;
unknown exact raster sizes cannot authorize downloads. No national dataset is
pulled implicitly. These commands discover small AOIs only:

```sh
python scripts/download_naip.py --bounds -122.43 37.76 -122.42 37.77 \
  --years 2020 2022 --region sf_example --max-items 2 --dry-run
python scripts/download_fema.py --bounds -122.43 37.76 -122.42 37.77 --dry-run
python scripts/download_hansen.py --bounds -122.43 37.76 -122.42 37.77 --dry-run
```

See [data pipeline](docs/data-pipeline.md) for local raster tiling, mask review,
manifest creation and dry-run size estimates. Source COG/granules may be much
larger than an AOI; estimated uncompressed bytes and actual transfer bytes are
reported separately. Data and checkpoints stay outside Git.

The loader accepts `sample/{pre,post,new_building,tree_removal}.png` or CSV paths
relative to the manifest, with `id,region_id,source,year_pre,year_post` and license
metadata. Output contains normalized `pre`/`post` tensors (3,H,W), concatenated
`image` (6,H,W) and integer `mask` (H,W): 0 background, 1 building, 2 tree removal.
Missing masks require explicit verified absence (`absent`); missing labels are
never silently treated as background during training. Optional auxiliary binary
masks and valid/review masks support future partial weak labels.

All geometric augmentation is shared by pre, post and labels: flips, rotate90,
small affine, translation and scale. Brightness/contrast, gamma, HSV, blur,
sharpening, JPEG and haze can vary independently by timestamp. Seeds replay
transforms. Synthetic building generators compare Telea, texture-copy and patch
fills with feathering; no black-rectangle deletion is used. Negative generators
cover exact copies, appearance/season/shadow/vegetation changes, slight shifts,
compression, blur and temporary objects. Real temporal negatives need verified
no-change provenance.

## Future training and validation

Training is implemented but has not been run. Default region splits preserve
connected spatial groups, adjacent tiles and shared source imagery. State,
temporal and random group assignments also retain geographic safety. One-region
data cannot claim independent validation. Reviewed full labels are required for
validation; partial labels outside valid/review masks use loss ignore index -100.

```sh
# After creating reviewed data/train_manifest.csv and retrieving baseline weights:
python scripts/train.py --config configs/train_short.yaml --dry-run --device cpu
# Future explicit training command (NOT executed during preparation):
python scripts/train.py --config configs/train_short.yaml
python scripts/validate.py --config configs/train_short.yaml \
  --checkpoint outputs/train_short/step_000025.pt --manifest data/approved/val.csv \
  --pred-dir outputs/val_probs
python scripts/search_thresholds.py --pred-dir outputs/val_probs \
  --ground-truth data/approved/val.csv --objective score --method staged
```

Training includes CUDA/CPU fallback, AMP, accumulation, encoder/head LRs,
step milestones, optimizer/scheduler/scaler/RNG/data-order resume, metrics and
optional early stopping. Losses: CE, weighted CE, Dice, CE+Dice, Focal,
Focal+Dice. Short preset compares 25/50/75/100 updates; medium extends to 1000.
See [experiment plan](docs/experiments.md) for resume semantics, first comparisons
and data leakage controls.

Local evaluation implements the published pair-presence macro F1 and tolerant
polygon shape aggregation, with explicit approximation choices for unpublished
private-scorer details. Pixel overlap and boundary metrics are diagnostics.
Threshold search varies class thresholds, component/total area and simplification
using the **exported geometry**, producing CSV trials and best YAML. See
[evaluation](docs/evaluation.md). Example thresholds and mock results are not
performance evidence.

## Polygon inference and submission

Reference conversion uses pixel-corner row-run boxes, Shapely union, component
area ≥30, total positive area ≥20, topology-preserving simplify 0.5 px and two
coordinate decimals. Optimized conversion coalesces equal runs vertically.
Configurable class-specific morphology, hole filling, component filtering and
polygon caps are supported. Submission exteriors cannot encode holes; this
policy is retained and scored consistently.

```sh
python scripts/predict_local.py --input-dir /path/to/input \
  --checkpoint /path/to/model.pt --config configs/inference_baseline.yaml \
  --output outputs/prediction.csv --prob-dir outputs/probabilities
python scripts/export_submission.py \
  --checkpoint 'baseline/original/03_illegal structure submission/assets/model/unet_r18_cd.pt' \
  --output outputs/submission_export
python scripts/make_submission_zip.py \
  --source outputs/submission_export --output outputs/submission.zip
```

ZIP root contains `predict.ipynb`, `requirements.txt`, `LICENSE`, `NOTICE`;
`assets/` holds model, config and shared offline inference code. The notebook
reads `AIF_INPUT_DIR`, finds `pairs.csv` and `images/<id>/{pre,post}.png`, and writes
`AIF_PREDICTION_PATH` with `id,new_building,tree_removal`. No change is an empty
CSV cell. There are no inference downloads, install magic or submit commands.
Competition submission and API keys remain the user's future manual operation.
See [submission guide](docs/submission.md).

## AWS preparation and repository map

[Future AWS setup](docs/aws-training.md) targets g5.2xlarge/A10G. The setup script
prints a plan by default; `--execute` installs dependencies on an already chosen
host, optionally syncs user-provided S3 URIs, and shows manual training/backup
commands. It never provisions or starts an instance or starts training.

```text
baseline/          immutable local organizer reference and hash inventory
configs/           baseline, short/medium training and inference presets
src/terradelta/     data, models, training, inference, polygons, metrics, external APIs
scripts/           discovery, preprocessing, training/validation/search, export, AWS setup
notebooks/         dataset preview and error analysis (no training)
tests/             CPU unit/integration and offline submission proof
docs/              baseline, pipeline, licenses, experiments, evaluation, AWS, submission
data/              local data only; see data/README.md
checkpoints/       protected model artifacts, ignored by Git
outputs/           local reports, predictions and ZIPs, ignored by Git
submission/template/  clean offline notebook template
```
