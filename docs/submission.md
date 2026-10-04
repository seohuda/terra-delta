# Offline submission package

Official specification: https://aifactory.space/ko/competitions/9306 .
The scoring service executes `predict.ipynb` at ZIP root, installs requirements
before inference, and blocks external inference network access. ZIP limit 6 GB;
installation up to 10 minutes and inference up to 120 minutes. Recheck live
rules before a future manual submission. No API key, debug submit, or actual
submit is implemented or executed here.

```sh
python scripts/export_submission.py \
  --checkpoint 'baseline/original/03_illegal structure submission/assets/model/unet_r18_cd.pt' \
  --config configs/inference_baseline.yaml --output outputs/submission_export
python scripts/make_submission_zip.py \
  --source outputs/submission_export --output outputs/submission.zip
```

Export rejects incompatible model metadata/keys and non-null pretrained weight
initialization. Training checkpoints are reduced to model weights and class /
encoder / step / seed provenance; optimizer/RNG state and local dataset paths
are not bundled. Original official inference checkpoints are copied byte-for-byte. The root has `predict.ipynb`, `requirements.txt`, `LICENSE`,
`NOTICE`; assets contain `model/model.pt`, `config.yaml`, and a vendored subset
of `terradelta` for shared offline behavior. This avoids a second divergent
inference implementation. No editable-install dependency, repository location,
weight download, training code, submit magic, or API key is required.
Requirements are torch, SMP 0.5.0, numpy, Pillow, Shapely and PyYAML; optional
alignment adds OpenCV. NumPy morphology needs no additional dependency.
Original attribution/license notices remain at root. Original baseline is never
modified; exports are a new destination and existing directories/ZIPs are not
overwritten.

Inputs: `AIF_INPUT_DIR` (`./input` default), exactly one recursively discovered
`pairs.csv` with `id` column and images relative to its parent:
`images/<id>/pre.png`, `images/<id>/post.png`. If the input has one ZIP, it can be
safely extracted locally. IDs stay strings, including leading zeros. Images
must be 256×256 RGB. Output: `AIF_PREDICTION_PATH` (`./prediction.csv` default),
columns `id,new_building,tree_removal`; CSV-quoted compact JSON polygon lists,
empty string for no change. Coordinates are pixel corners in the post image,
x right, y down. Only exteriors are exported; holes are filled by representation.

Run from the extracted submission root as in the official baseline. Notebook
uses only local assets and Python (no shell/magic commands). CPU works; CUDA
readiness is probed before inference and may fall back to CPU. Local pytest
executes all generated code cells against a tiny mock input with network calls
blocked, using actual official weights; outputs are compared to baseline CSV.

Raw probability maps are available through local CLI `--prob-dir` before
postprocessing; each `.npy` is float32 3×256×256, class order 0/1/2. They are not
submission requirements. The baseline preset disables TTA and alignment.
Conservative preset's 0.55/0.60 thresholds are unvalidated examples.
