# Official baseline analysis

Inspected 2026-10-04: actual Google Drive archive, notebook JSON, requirements,
LICENSE, NOTICE, and `torch.load(..., weights_only=True)` checkpoint. Source:
https://drive.google.com/file/d/1RZjoG0hITfY5XqIPuv5KtKYV01XUiOT3/view .
SHA-256 inventory: `baseline/integrity.json`. The original is unchanged locally.

```
03_illegal structure submission/
  LICENSE                          5,970 bytes
  NOTICE                           2,469 bytes
  requirements.txt                   259 bytes
  predict.ipynb                   13,586 bytes
  assets/model/unet_r18_cd.pt  57,461,007 bytes
```

## Model and inputs

Notebook cell 4 instantiates SMP `Unet(encoder_name="resnet18",
encoder_weights=None, in_channels=6, classes=3)`. It loads checkpoint
`state_dict` strictly on CPU before moving the model to the inference device.
Class order is background, new_building, tree_removal. The checkpoint metadata
is `classes=["background","new_building","tree_removal"]`, `in_channels=6`,
`encoder="resnet18"`, `steps=100`, `seed=0` (verified from the actual checkpoint).
No optimizer or training dataset is bundled. The 100-step metadata does not
establish that 100 steps is optimal on any new dataset.

RGB bytes are read via Pillow/BytesIO, converted to RGB, and rejected unless
256×256. Pre and post are independently divided by 255, normalized with mean
[0.485,0.456,0.406] and std [0.229,0.224,0.225], concatenated HWC (pre first),
cast float32, stacked, and permuted to B×6×256×256. Cell 4 applies softmax on
channel axis then argmax. Batch size is 16. Model eval and no_grad are used.
Device order is CUDA, MPS, CPU; one zero-image accelerator trial runs, with CPU
fallback on failure. TerraDelta's local proof explicitly uses CPU.

## Polygon conversion

Cell 1 defines H=W=256, MIN_AREA=30, MIN_POS_AREA=20.0, SIMPLIFY_PX=0.5,
NDIGITS=2. For each nonempty row, contiguous runs become Shapely boxes at pixel
corners: pixel (r,c) occupies [c,c+1]×[r,r+1]. `unary_union` creates geometry;
MultiPolygon parts are separated, polygons below 30 area discarded, then total
remaining area below 20 returns empty. Parts simplify with topology preserved.
Only exterior coordinates are serialized, rounded to two decimals, with the
closing duplicate vertex removed. Compact JSON polygon lists go in CSV cells;
no polygons means empty string. Holes are therefore filled by the submission
representation even if original masks contain holes. Reference parity is based
on this exact ordering, including area checks before simplification/exteriors.

The verbatim reference function is retained with source attribution separately
from configurable and optimized postprocessing. The latter rechecks emitted
polygon area for its configured presence threshold.

## I/O and execution

`AIF_INPUT_DIR` defaults to `./input`; `AIF_PREDICTION_PATH` defaults to
`./prediction.csv`. The notebook recursively finds `pairs.csv`, choosing the
first sorted match. If none is found and exactly one ZIP exists below input,
it extracts and searches it. Nonblank ids are stripped from CSV (UTF-8 BOM
supported). Files come from the selected CSV's parent:
`images/<id>/pre.png`, `images/<id>/post.png`. Directory scanning does not
invent ids. Output columns are exactly `id,new_building,tree_removal`; stdlib
CSV quoting protects JSON cells; output parent directories are created.

TerraDelta preserves this layout and recursive/ZIP discovery, but rejects
ambiguous manifests, duplicate ids, path traversal, ZIP symlinks, and excessive
ZIP expansion instead of trusting malformed inputs. Assets are local relative
to submission root. Original final cell contains `%pip install aifactory`,
`%load_ext aifactory`, and an actual submit command with a placeholder key.
It is never executed here and is omitted from generated notebooks.

## Dependencies and licenses

Actual requirements: torch>=2.5, segmentation-models-pytorch>=0.5,
numpy>=2.0, pillow>=10.0, shapely>=2.0. No notebook inference install cell is
needed. TerraDelta pins SMP 0.5.0 for checkpoint compatibility; extra packages
are included in exports only when configured features require them.

LICENSE grants use, modification, retraining and redistribution for competition
participation and says outside-competition use follows contest rules. It includes
torchvision BSD-3-Clause, SMP MIT, Shapely BSD-3-Clause texts and describes
GEOS dynamic LGPL-2.1 use. NOTICE identifies ResNet18 ImageNet encoder weights
from `smp-hub/resnet18.imagenet`, fine-tuned with organizer materials. These
organizer grants are distinct from a general claim that every ImageNet-derived
weight is suitable for unrestricted commercial use; see LICENSE_DATA.md.

## Observed limitations and planned comparisons

1. Softmax argmax only; no class-specific threshold.
2. No TTA.
3. No alignment correction (input described as already aligned).
4. No independent class postprocessing parameters.
5. No uncertainty handling.
6. No confidence-based empty prediction rule.
7. No separate hard-negative suppression.
8. No dataset/training/validation implementation in this submission archive.

TerraDelta supplies configurable class thresholds/morphology, optional TTA and
bounded translation/ECC alignment, hard-negative generators, region splits,
local geometry evaluation, and short optimizer-step checkpoint schedules.
Default baseline mode remains a reproducible reference; thresholds in example
configs are hypotheses, never measured optima. No competition score is claimed.
