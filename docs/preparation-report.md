# TerraDelta preparation report

Completed code and CPU verification on 2026-10-04. The project is TerraDelta,
repository terra-delta. This report describes preparation, not model performance.

## Changed files

README, packaging and dependency files; configs; immutable-baseline provenance;
data/model/training/inference/postprocess/evaluator modules; downloader,
preprocessing, validation/search, local prediction and ZIP scripts; CPU tests;
preview/error notebooks; licenses and AWS/submission/experiment documentation.
The complete tracked file list is appended below. Data, weights, ZIPs, outputs,
environments and credentials are not committed.

## Test results

- Full suite: **421 passed, zero failed/skipped**, 26.36 seconds, CPU only.
- 51 pending-deprecation warnings come from affine/rasterio's multiplication API;
  no warning indicates a failed loader, malformed output or trained model.
- Ruff, Python compile/import checks, notebook schema validation, git diff check
  and AWS shell syntax/plan-only invocation passed.
- Original baseline file sizes/SHA-256 match baseline/integrity.json unchanged.
- Official checkpoint strict load and forward: finite [1,3,256,256]; original
  notebook preprocessing/argmax matches TerraDelta on the same pair.
- Exported ZIP is executed in a fresh CPU process with outbound network blocked;
  baseline and conservative preset CSVs match the local pipeline. Generated
  notebook was also executed in a separate Jupyter CPU kernel on four mock pairs.
- Loader/augmentation/geometry/splits, partial-label ignore masks, no-clobber
  locks, six losses, RNG/data-order/checkpoint restore, validation/search and
  streamed download failure guards are tested without any optimizer updates.
- Actual NAIP/FEMA/Hansen metadata-only CLIs succeeded; no image/footprint payload
  was fetched. Local logs: outputs/pytest-final.txt and outputs/*-dry-run.json.

## Facts established from the actual baseline

SMP UNet, ResNet18, six input channels, three classes in order background /
new_building / tree_removal. Pre and post 256×256 RGB are normalized separately
with ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225] before pre-first
concat. Checkpoint has 182 tensors, steps=100, seed=0, and 14,337,907 model parameters.
Inference is softmax argmax, batch16. Row-run pixel boxes → union → component
area30 → total area20 → topology-preserving simplify0.5 → exterior-only JSON with
two decimals. Official notebook's actual submit cells were inspected, never run.
Original permission is for competition participation; notices remain in exports.
See docs/baseline-analysis.md for the full source-based analysis.

## Intentionally not executed / remaining real-world limits

No GPU/CPU training, optimizer update, AWS resource provisioning/start/stop,
S3 transfer, paid resource creation, large NAIP/FEMA/Hansen acquisition, real
training dataset generation, AIFactory debug/actual submission or API-key use.
GPU AMP and actual optimizer-step convergence are untested by design. Resume
state restoration is tested without fitting. No real validation/competition score
or best checkpoint is claimed. Local evaluator is an explicit approximation of
published rules; private geometry/zero-denominator behavior is not verified.
FEMA terms say CC BY4 while a badge points to BY3, so discovery reports unknown
and training excludes it until exact attribution/version review. Real polygon
pagination and downloaded imagery/label quality remain future checks.

## Commands for future training

After activating the environment, retrieving the unchanged baseline and creating
reviewed data/train_manifest.csv with at least two disconnected spatial groups:

```sh
source .venv/bin/activate
python scripts/train.py --config configs/train_short.yaml --dry-run --device cpu
# Future explicit training, not executed in preparation:
python scripts/train.py --config configs/train_short.yaml
python scripts/train.py --config configs/train_short.yaml --resume outputs/train_short/last.pt
```

The preset initializes from the official local weights and compares additional
optimizer updates 25/50/75/100. Resume needs the same compatible run/config/data;
for a different budget use an explicit warm-start in a new output directory.

## Commands for future AWS startup

Choose an already reviewed project instance/profile/region; placeholders are not
account defaults. No cloud calls below were executed:

```sh
export AWS_PROFILE=YOUR_PROFILE
export AWS_REGION=YOUR_REGION
export TERRADELTA_INSTANCE_ID=YOUR_PROJECT_INSTANCE_ID
aws ec2 start-instances --instance-ids "$TERRADELTA_INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$TERRADELTA_INSTANCE_ID"
aws ec2 wait instance-status-ok --instance-ids "$TERRADELTA_INSTANCE_ID"
# SSH to the chosen host and cloned repository:
bash scripts/aws_setup.sh                    # plan only
bash scripts/aws_setup.sh --execute          # setup, still no training
```

See docs/aws-training.md for AMI/driver/role choices, S3 backup and stopped-state
verification. g5.2xlarge/A10G is the planned host, not a created or running instance.

## Expected future data volume

HEAD-confirmed examples, not downloads:

| Asset/representation | Bytes | Scope |
| --- | --- | --- |
| Maryland NAIP 2021 COG | 447,829,958 (~448 MB) | Full source tile, 0.6 m, 12,390×9,860×4 |
| Same NAIP raw imagery | 488,661,600 | Uncompressed estimate, not transfer size |
| Hansen v1.12 lossyear 40N 080W | 42,217,473 (~42 MB) | Full 10-degree granule |
| FEMA chosen AOI | 218 features, ~327,000 bytes | Geometry-size heuristic; no exact payload size |
| One 256×256 RGB temporal pair + two uint8 masks | 524,288 | Raw storage, PNG compression varies |

1,000 prepared pairs plus masks are about 524 MB raw; 10,000 about 5.24 GB raw,
excluding original rasters, metadata, float probability maps and checkpoints.
Actual acquisition totals depend on AOIs/years, complete discovery and HEAD.
The NAIP example exceeds default guards and will not download implicitly. Small
AOI tensor estimates cannot substitute for whole-source COG transfer sizes.

## First future experiment

First evaluate unchanged official weights on geographically disjoint, fully
reviewed licensed data in exact baseline mode; save probabilities and inspect
false positive/missed positive pairs. Then compare one seed-0 short fine-tune
at 25/50/75/100 additional steps, holding the split fixed. Tune per-class
threshold/component/total-area/simplification only on validation, with a separate
untouched test geography. Compare TTA, alignment and synthetic/negative variants
only after the reference comparison, one change at a time. No measured optimum
or real score currently exists.

## Local review artifacts

- outputs/submission.zip: offline package using unchanged organizer weights,
  not a trained TerraDelta model or a submitted entry.
- outputs/submission_export/: root-layout notebook, requirements, notices,
  assets/model/model.pt, assets/config.yaml, vendored shared inference code.
- outputs/smoke/report.json, training_dry_run.json, executed_predict.ipynb:
  four small mock pairs, optimizer_steps_executed=0, matching CSV output.
- outputs/*-dry-run.json: metadata/HEAD plans without raster/footprint payload.

## Git delivery

Verified implementation is committed in meaningful stages and prepared for push
to https://github.com/seohuda/terra-delta . Remote completion is checked after push.

## Tracked changed-file inventory

```
.gitignore
LICENSE_DATA.md
README.md
baseline/README.md
baseline/integrity.json
baseline/original/.gitkeep
checkpoints/.gitkeep
configs/baseline.yaml
configs/inference_baseline.yaml
configs/inference_conservative.yaml
configs/train_medium.yaml
configs/train_short.yaml
data/README.md
docs/aws-training.md
docs/baseline-analysis.md
docs/data-pipeline.md
docs/evaluation.md
docs/experiments.md
docs/licenses.md
docs/preparation-report.md
docs/submission.md
notebooks/dataset_preview.ipynb
notebooks/error_analysis.ipynb
outputs/.gitkeep
pyproject.toml
requirements-dev.txt
requirements.txt
scripts/aws_setup.sh
scripts/benchmark_polygons.py
scripts/build_dataset.py
scripts/download_fema.py
scripts/download_hansen.py
scripts/download_naip.py
scripts/export_submission.py
scripts/inspect_baseline.py
scripts/make_submission_zip.py
scripts/predict_local.py
scripts/search_thresholds.py
scripts/smoke_test.py
scripts/train.py
scripts/validate.py
src/terradelta/__init__.py
src/terradelta/data/__init__.py
src/terradelta/data/dataset.py
src/terradelta/data/forest_labels.py
src/terradelta/data/negatives.py
src/terradelta/data/pairing.py
src/terradelta/data/split.py
src/terradelta/data/synthetic_building.py
src/terradelta/data/transforms.py
src/terradelta/external/__init__.py
src/terradelta/external/common.py
src/terradelta/external/fema_structures.py
src/terradelta/external/hansen.py
src/terradelta/external/naip.py
src/terradelta/inference/__init__.py
src/terradelta/inference/alignment.py
src/terradelta/inference/predictor.py
src/terradelta/inference/tta.py
src/terradelta/metrics/__init__.py
src/terradelta/metrics/competition.py
src/terradelta/metrics/evaluation.py
src/terradelta/metrics/segmentation.py
src/terradelta/models/__init__.py
src/terradelta/models/checkpoint.py
src/terradelta/models/factory.py
src/terradelta/models/unet_r18.py
src/terradelta/postprocess/__init__.py
src/terradelta/postprocess/mask.py
src/terradelta/postprocess/polygons.py
src/terradelta/postprocess/reference.py
src/terradelta/postprocess/thresholds.py
src/terradelta/submission.py
src/terradelta/training/__init__.py
src/terradelta/training/callbacks.py
src/terradelta/training/losses.py
src/terradelta/training/scheduler.py
src/terradelta/training/search.py
src/terradelta/training/splits.py
src/terradelta/training/trainer.py
src/terradelta/training/validation.py
src/terradelta/utils/__init__.py
src/terradelta/utils/geo.py
src/terradelta/utils/io.py
src/terradelta/utils/logging.py
src/terradelta/utils/seed.py
submission/template/README.md
submission/template/predict.ipynb
submission/template/requirements.txt
tasks/todo.md
tests/conftest.py
tests/test_alignment.py
tests/test_checkpoint.py
tests/test_config.py
tests/test_dataset.py
tests/test_external.py
tests/test_metrics.py
tests/test_model.py
tests/test_polygon.py
tests/test_postprocess.py
tests/test_search.py
tests/test_submission.py
tests/test_synthetic.py
tests/test_training.py
tests/test_workflow_safety.py
```
