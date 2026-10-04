# Local data workspace

Keep downloaded assets and generated datasets local. `.gitignore` excludes
`data/**` except this README. It also excludes checkpoints and archives.
Never commit external imagery, competition data, SAS URLs, credentials or weights.

Suggested layout (directories are created only when a later operation needs them):

```text
data/
  raw/naip/                 # unchanged downloaded GeoTIFF/COG assets
  raw/fema/                # bounded AOI GeoJSON and provenance
  raw/hansen/              # version-pinned candidate-mining granules
  candidates/<region>/     # local pre/post tiles, no implicit change labels
  reviewed/<sample-id>/     # pre/post RGB and class masks, optional scope masks
  manifests/               # candidates, reviewed, train, val and exclusion CSVs
```

Each annotated sample has `pre.png`, `post.png`, `new_building.png` and
`tree_removal.png`. RGB pairs have equal dimensions; class masks are single-channel
0/1/255 with no overlap. Missing labels cannot imply no change. In a manifest,
use `absent` or a `<class>_absent=true` field only for a reviewed absent class.

Manifest paths resolve relative to the CSV. Preserve `region_id`, state, sources,
years/acquisition dates, source URLs/version, license evidence/status, label review
status, reviewer, bounds, CRS and parent/pair IDs. Geographic split defaults to
whole regions and joins neighbors/shared imagery. Temporal split needs an explicit
`temporal_year` and holds out the entire connected geography of a recent sample.

Optional `valid_mask` and `review_mask` paths define labeled scope. Only their
intersection is trusted: outside pixels become `-100`, never background negatives.
Partial scopes are training-only; validation needs fully annotated ground truth.
`has_labels` alone does not assert full coverage. Optional `auxiliary_mask` output
contains two Boolean class channels and requires the returned validity mask.

Current verification status: tiny temporary fixtures and live metadata checks only;
no imagery/footprint acquisition, generated real training dataset, training,
optimizer steps, AWS actions or submission. Do not include Hansen's coarse loss
grid as segmentation truth. Candidate refinement requires high-resolution review.

See [pipeline commands and limits](../docs/data-pipeline.md),
[permission inventory](../LICENSE_DATA.md) and [source audit](../docs/licenses.md).
