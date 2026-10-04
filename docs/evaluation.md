# Evaluation

## Published polygon score

`terradelta.metrics.evaluate_predictions(pred_rows, gt_rows)` implements the
[competition's published evaluation rules](https://aifactory.space/ko/competitions/9306)
(evaluation section, lines 156–187, checked 2026-10-04). It is a local
approximation, not a claim of bit-for-bit parity with the private scorer.
`terradelta.metrics.competition` reexports that same function, `CHANGE_CLASSES`,
and `MIN_PREDICTION_AREA` for compatibility; implementation lives in `evaluation.py`.

Inputs are iterable mappings with required `id`, `new_building`, and
`tree_removal` fields. IDs must be unique, hashable, nonempty and finite; the
prediction and GT ID sets must match exactly. Order is irrelevant and no string
coercion is performed. Duplicate IDs, missing columns, missing/extra IDs, or
invalid geometry raise `ValueError` with context. Pass CSV `DictReader` rows or
dataframe `to_dict("records")`, not a dataframe itself.

Each class cell contains a list of polygon exterior rings, or a JSON string
encoding that list: `[[[x, y], [x, y], ...], ...]`. Each ring has at least three
points with finite numeric XY coordinates and must define a valid positive-area
polygon. The original notebook omits the repeated closing point; Shapely closes
these rings automatically. Already closed rings are also accepted. `None`, float
NaN, blank/whitespace CSV strings, the CSV string `nan` (case insensitive), `[]`,
JSON `[]`, and JSON `null` mean no polygons. Malformed/nested non-finite JSON,
self-intersections, degenerate rings, holes/GeoJSON encodings, and nonnumeric
coordinates are rejected. No `make_valid`, zero-distance repair buffer, snapping,
clipping, rasterization, or simplification is performed.

For each class independently:

1. Union all supplied exteriors on each side. Overlapping polygons are counted
   once. Both scoring layers use these same unions, including any holes formed
   naturally by a union. Each input ring itself is an exterior, not a hole.
2. GT is positive whenever polygons are present, regardless of their area.
   Prediction is positive when its **union area is at least 20** square pixels.
   A union area below 20 is an empty prediction for both presence and shape.
3. Compute positive-label `F1 = 2 TP / (2 TP + FP + FN)` and negative-label
   `F1 = 2 TN / (2 TN + FP + FN)`. Presence macro F1 is their unweighted mean.
4. On **every GT-positive pair**, compute
   `precision = area(pred ∩ dil1(gt)) / area(pred)` and
   `recall = area(gt ∩ dil1(pred)) / area(gt)`, then their harmonic mean.
   `dil1(g)` is `g.buffer(1, join_style="mitre")`. Missing/subthreshold predictions
   score zero and stay in the average. Shape is an unweighted mean over pairs,
   not a GT-area-weighted mean and not an average restricted to true positives.
5. Class score is `0.5 * presence_macro_f1 + 0.5 * shape_score`; the final score
   is the mean of the two class scores. Overlap between change classes is allowed.

Explicit approximation choices: every undefined/zero-denominator F1 or ratio is
zero; a class with no GT-positive pairs has shape score zero; an entirely empty
dataset scores zero. The published text does not settle absent-label F1 behavior,
so both binary labels are always retained. Consequently, an all-negative perfect
dataset has presence macro F1 0.5 and class score 0.25; an all-positive perfect
dataset has presence macro F1 0.5 and class score 0.75. Do not mistake those for
the mixed-positive/negative perfect score of 1.

Areas are continuous planar polygon areas in pixel-coordinate units, not counts
of rasterized pixels. Shapely's default mitre limit of 5 is retained; unusually
acute corners may be beveled by that limit. The private scorer's geometry engine,
floating-point precision, and corner-limit settings are not published.

```python
from terradelta.metrics import evaluate_predictions

square = [[0, 0], [10, 0], [10, 10], [0, 10]]  # unclosed exterior
rows = [
    {"id": "positive", "new_building": [square], "tree_removal": [square]},
    {"id": "negative", "new_building": "", "tree_removal": ""},
]
scores = evaluate_predictions(rows, rows)
assert scores["score"] == 1.0
```

The result is a JSON-serializable dictionary:

| Key | Meaning |
| --- | --- |
| `score` | Final mean class score |
| `num_samples` | Number of matched IDs |
| `classes[class_name].presence_macro_f1` | Presence macro F1 over both labels |
| `classes[class_name].shape_score` | Mean tolerant shape F1 over all GT-positive pairs |
| `classes[class_name].score` | Equal-weight presence/shape class score |
| `classes[class_name].gt_positive_count` | Shape denominator |
| `classes[class_name].presence_confusion` | `tp`, `fp`, `fn`, `tn` pair counts |

## Raster diagnostics

These diagnostics do not replace the published polygon evaluator or apply its
20-pixel cutoff. All undefined ratios are zero; there is no smoothing constant.

| API exported by `terradelta.metrics` | Input and output |
| --- | --- |
| `binary_segmentation_metrics(prediction, ground_truth)` | Equal-shaped nonscalar boolean/0–1 arrays; pooled `tp`, `fp`, `fn`, `tn`, `iou`, `dice`, `f1`, `precision`, `recall`, `accuracy` |
| `multiclass_segmentation_metrics(prediction, ground_truth, num_classes, *, ignore_index=None)` | Exclusive integer labels in `[0, num_classes)`; GT-row/prediction-column `confusion_matrix`, integer-keyed `per_class` binary metrics, `macro_iou`, `macro_dice`, `macro_f1`, pixel `accuracy`, `num_pixels` |
| `boundary_metrics(prediction, ground_truth, *, tolerance=1.0)` | Nonempty 2-D binary masks; directional boundary `precision`, `recall`, `f1`, total and matched boundary-pixel counts |
| `multiclass_boundary_metrics(prediction, ground_truth, num_classes, *, tolerance=1.0)` | Exclusive 2-D integer labels; integer-keyed `per_class` boundary metrics and `macro_f1` |

Overlap functions pool batches if passed batched arrays. Multiclass macros
include all declared classes, including background and absent classes; absent
classes get zero overlap F1/IoU. `ignore_index` excludes GT pixels from every
overlap count, including any prediction at those pixels. Floats/probabilities
are rejected as multiclass labels. For change channels that overlap, evaluate
each channel independently with binary functions; argmax loses that information.

Raster boundaries are foreground pixels minus a full 3×3 erosion, with outside
the image treated as background. This includes image-edge and hole boundaries.
Precision measures predicted boundary pixels within `tolerance` of any GT
boundary pixel; recall measures the reverse direction. Euclidean pixel distances
are used, with many-to-many matching, followed by harmonic F1. At tolerance 1,
diagonal single-pixel offsets are not matches. This differs from the polygon
score's mitre-buffered **area** overlaps, which admit the expanded square corners.
Empty boundaries score zero even when both masks are empty. Call boundary metrics
per image, not on a batched volume. Multiclass boundaries have no ignore-index
option because masking ignored pixels would create artificial boundaries.

## API sources and verification

Geometry uses the official Shapely APIs:
[Polygon](https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html),
[union_all](https://shapely.readthedocs.io/en/stable/reference/shapely.union_all.html),
and [buffer](https://shapely.readthedocs.io/en/stable/reference/shapely.buffer.html).
Presence F1 is calculated directly from counts with the documented
[F1 formula](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html),
retaining both labels and choosing zero for undefined ratios without requiring
scikit-learn. Raster confusion uses
[NumPy bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html).
Raster boundaries use SciPy
[binary_erosion](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html)
and [distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html).
SciPy is imported lazily by boundary diagnostics; polygon/overlap evaluation needs
only the project's core NumPy and Shapely dependencies.
SciPy is required only when calling those boundary diagnostics (it is available
in the local environment); postprocess morphology, including enhanced modes,
uses NumPy and does not require SciPy.

Targeted verification (existing environment only):

```sh
.venv/bin/python -m pytest tests/test_metrics.py
```

Tests cover hand-calculated layer/class weights, positive and negative presence
F1, threshold equality, union versus summed area, GT-only shape averaging,
directional shape precision/recall, square-corner dilation, CSV empties, strict
validation, raster confusion/ignore behavior, and Euclidean boundary matching.
No training or optimizer steps are involved.

## Postprocess source and configuration

The verbatim official constants and `mask_to_polygons` function live in
`src/terradelta/postprocess/reference.py`, extracted from setup cell 1 of
`baseline/original/03_illegal structure submission/predict.ipynb`. This source
function returns the original compact polygon JSON string or `''` for an empty
prediction. Its conversion body/constants are preserved unchanged.

Public adapters in `terradelta.postprocess.polygons`:

| Function | Contract |
| --- | --- |
| `mask_to_polygons_reference(mask, min_area=30, min_pos_area=20, simplify_px=0.5, ndigits=2)` | JSON string or `''`; delegates to the original source on default settings, with a parameterized reference path for overrides |
| `mask_to_polygons(mask, ..., backend="runs", max_polygons=None)` | Parsed exterior lists; optimized run coalescing by default; supports explicit reference and optional rasterio backends |
| `predictions_to_polygons(probabilities, config=None)` | `(3,H,W)` probabilities ordered background/new_building/tree_removal to a dictionary of exterior lists keyed `new_building` and `tree_removal` |
| `serialize_polygons(polygons)` | Compact JSON string for a list of exteriors, or `''` for an empty list |

Default `predictions_to_polygons` mode is `argmax`, backend `reference`, with
no morphology, polygon cap, or connected-component prefilter. This defaults to
the original conversion, though the adapter returns parsed lists rather than
JSON strings. An entire config containing `postprocess` or a direct postprocess
mapping is accepted. Common settings are inherited by each class; overrides in
`classes.new_building` / `classes.tree_removal` apply independently, followed by
any direct top-level class overrides. Nested morphology settings merge per key.

The following is a schema example with class-specific thresholds, not tuned
parameters or a performance recommendation:

```yaml
postprocess:
  mode: threshold             # argmax is the reference default
  backend: runs               # optimized is an alias; reference or optional rasterio also supported
  ndigits: 2
  classes:
    new_building:
      threshold: 0.5
      min_area: 30
      min_pos_area: 20
      simplify_px: 0.5
      ndigits: 2
      max_polygons: null
      cc_min_area: 0
      connectivity: 4
      morphology:
        opening: 0
        closing: 0
        dilation: 0
        erosion: 0
        fill_holes: false
    tree_removal:
      threshold: 0.5
      min_area: 30
      min_pos_area: 20
      simplify_px: 0.5
      ndigits: 2
      max_polygons: null
      cc_min_area: 0
      connectivity: 4
      morphology:
        opening: 0
        closing: 0
        dilation: 0
        erosion: 0
        fill_holes: false
```

| Per-class setting | Meaning |
| --- | --- |
| `threshold` | Finite `[0,1]` value; threshold mode admits pixels at or above the class threshold, then selects the highest probability among background and eligible positives; background wins ties and building wins positive ties |
| `min_area` | Component area cutoff; reference filters before simplification/exterior export, optimized also checks emitted rounded exterior area; default 30 square pixels |
| `min_pos_area` | Minimum retained total area; reference checks before export, optimized also checks the emitted exterior union; default 20 square pixels |
| `simplify_px` | Nonnegative topology-preserving simplification tolerance in pixels; default 0.5 |
| `ndigits` | Nonnegative integer coordinate decimal precision; default 2 |
| `max_polygons` | Optional nonnegative cap; largest polygons retained with stable bounds tie-break, before total-area filtering; null disables |
| `cc_min_area` | Foreground connected-component pixel-count filter before polygon conversion; zero disables |
| `connectivity` | 4 or 8 for the mask component filter; polygon geometry still follows pixel-corner unions |
| `morphology.opening/closing/dilation/erosion` | Nonnegative integer radius of a square `(2r+1)` kernel; zero disables |
| `morphology.fill_holes` | Boolean; fill background components disconnected from the image border using 4-way connectivity |

Mask operations run opening → closing → dilation → erosion → optional hole fill
→ component filter, independently per class. Pixels outside the image are
background. Polygon coordinates use pixel corners: raster `(row, col)` occupies
`[col,col+1] × [row,row+1]`. Exported rings omit the repeated closing coordinate
and omit holes, so evaluating original masks instead of exported exteriors can
give different results. The evaluator's mandatory union-area cutoff is applied
after export, even if configurable postprocess thresholds differ. The optimized
backend can differ in vertex ordering/serialization while preserving geometry;
use `reference` when exact original serialization is required.
Optimized export additionally enforces `min_area` on rounded exterior polygons
and `min_pos_area` on their final union. Reference export preserves the original
pre-simplification/pre-hole-removal filtering exactly. These behaviors can differ
near cutoffs, after rounding, or when exterior export fills holes.

<!-- Parallel postprocess implementation may append further API/configuration details here. -->
