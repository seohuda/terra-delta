# Future experiments — no training results yet

Preparation has executed no optimizer update, backward training run, cloud GPU,
large source imagery download, or AIFactory submit. Mock scores validate code
paths only; they say nothing about detection quality. Official metadata says
100 optimizer steps in the organizer checkpoint, not that 100 is an optimum for
external data or the hidden evaluation set.

## First comparison

1. Select two or more **disconnected** licensed NAIP regions and complete,
   reviewed 256×256 labels. FEMA footprints are static building references, not
   evidence of new construction. Hansen finds candidate areas only. Keep all
   derivatives, adjacent tiles, and years from one spatial group on one side.
   Freeze region split and keep an independent test geography for final reporting.
2. Evaluate unchanged official weights in exact baseline mode first (no TTA,
   alignment or class thresholds). Save raw probabilities, classification errors,
   tolerant polygon shape scores, and example failures. This is the real local
   reference; no score has been measured on real data in preparation.
3. Compare a single short fine-tuning run at steps 25/50/75/100, seed 0. The short
   preset initializes from the official local checkpoint, uses CE+Dice with
   background downweighting and separate encoder/head LRs. Include reviewed
   negatives as well as building/tree examples; never label an arbitrary real
   pair background solely because it has similar season or location.
4. On the same validation geography, search per-class probabilities and area /
   simplification settings with saved maps. Fit these only on validation, never
   on the independent test geography. Examine negative FP rates and missed GT
   positive pairs, because both layers penalize false/missed presence.
5. The first upper-budget comparison uses `configs/train_first250.yaml` with
   checkpoints **25/50/75/100/150/250**, seed 0, encoder LR 1e-5 and head LR 1e-4.
   Inspect 25–100 first; 150/250 are comparisons, not assumed improvements.
   Do not prioritize the medium 500+ step experiment before these checks.
   At every checkpoint compare approximate local score, each class score,
   FP/FN, no-change FP rate and confidence distributions on the same split.
   Compare synthetic fill strategies,
   hard-negative sampling and TTA one change at a time. Optional alignment needs
   a known-shift sanity check and a validation ablation before use on real data.

The preset thresholds (0.55 building, 0.60 tree) are unvalidated hypotheses.
Choose the checkpoint and thresholds by the published local score AND inspect
class-specific behavior. The local evaluator is an approximation; no private
scorer parity or competition score is claimed.

## Future commands

From the repo root, activate the installed environment. Fill real manifest paths
in a copied config or `data/train_manifest.csv` with source/license metadata.
Default region split is enforced by the same graph as `build_dataset`.
Explicit train/val manifests are supported through `data.train_manifest` and
`data.val_manifest`; same regions, common source images, adjacent/buffered bounds,
state groups (for state split), and duplicate ids are rejected across partitions.
A temporal split additionally requires `data.split.temporal_year`; whole spatial
groups containing images at/after that year go to validation. Older images of
the held-out geography cannot leak into training.

```sh
python scripts/train.py --config configs/train_short.yaml --dry-run --device cpu
# Future training, deliberately NOT executed during preparation:
python scripts/train.py --config configs/train_short.yaml
python scripts/train.py --config configs/train_short.yaml \
  --resume outputs/train_short/last.pt
python scripts/validate.py --config configs/train_short.yaml \
  --checkpoint outputs/train_short/step_000025.pt --pred-dir outputs/val_probs
python scripts/search_thresholds.py --pred-dir outputs/val_probs \
  --ground-truth data/approved/val.csv --objective score --method staged
```

Pass the **same validation partition** to search as the one that produced maps;
id sets must match exactly. To make that explicit, validate with `--manifest
 data/approved/val.csv` and search that same file. `--ground-truth-kind polygons`
accepts exact exterior CSV labels, `manifest` loads masks. `--objective score`
selects the final local competition score; a dotted class score is also allowed.
Search output defaults to `outputs/threshold_search.csv` and
`outputs/best_postprocess.yaml`. Cartesian search is available but can be large;
staged coordinate search keeps the incumbent and is not globally optimal.

## Training semantics and reproducibility

`max_optimizer_steps` counts successful AdamW updates, not epochs or microbatches.
Default effective batch is 4×4=16, except partial accumulation at an epoch's end.
The CLI holds an exclusive output-directory lock and rejects a new run that would
overwrite existing checkpoints/metrics. Use an explicit resume or a new directory.
Those groups are normalized by their actual sample count and do not cross epochs.
Milestones are config-controlled. Failed AMP overflow updates do not increment
steps and have a bounded failure counter. CUDA autocast is float16; CPU fallback
uses bfloat16 when AMP is enabled. Loss reductions stay float32. Unavailable
CUDA falls back to CPU; a long CPU run is a future explicit user operation only.

Checkpoints contain strict model state, class/encoder metadata, optimizer,
scheduler, scaler, RNG streams, epoch permutation and next-batch offset, config
signature, metrics/early-stop state, runtime versions. Resume restores all these
at optimizer boundaries. Configuration budget, split, manifests, batch/accumulation,
LR/loss and runtime/device must remain compatible. For a different step budget
or runtime, use a local `--init-checkpoint` as a **new** experiment, with a new
output directory; that warm-start resets optimizer/scheduler and is not exact
resume. Default encoder_weights is null, and train CLI rejects internet encoder
initialization. Use local reviewed weights only.

A single synchronous loader (`num_workers=0`) and deterministic per-sample seed
provide a clear mid-epoch resume boundary. Parallel workers/prefetch are rejected
because hidden queues/RNG would invalidate this guarantee. Treat the source
images/labels as immutable throughout a run. Record source/licensing and protect
all checkpoint files before any future cleanup. Partial weak labels need
valid/review masks and loss ignore index -100; the validation set must have full
annotations. Metadata approval strings are review records, not automatic legal
permissions; unknown/forbidden sources remain excluded by default.

Metrics append to `metrics.jsonl`. Optional early stopping observes an explicit
metric and patience in validation events; `best.pt` tracks improvements when a
metric is configured even if early stopping is disabled. `last.pt` supports
recovery; every requested step checkpoint supports direct comparison. For actual
GPU throughput, profile the permitted short run later; no GPU timings exist yet.

## Non-training proof

```sh
python -m pytest -q
python scripts/smoke_test.py --output outputs/smoke_new --execute-notebook
python scripts/benchmark_polygons.py --repeats 3
```

The smoke command generates four small mock RGB/mask pairs, runs train **dry-run**
(no optimizer construction), validation, probability search, local exporter and
ZIP builder. With `--execute-notebook`, a separate local CPU kernel executes the
generated notebook with outbound Python network operations blocked. It compares
CSV bytes to local inference. Existing evidence directories are never overwritten.
Benchmark compares actual GEOS conversion timings on synthetic 256×256 masks;
optimized and reference may reorder vertices but geometric equality is checked
with simplification disabled. This benchmark does not assert a fixed speedup.
