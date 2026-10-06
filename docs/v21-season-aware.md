# v2 MAIN and bounded v2.1 continuation

## Frozen v2 MAIN

One unchanged v2 ZIP was submitted to MAIN on 2026-10-04 at 14:07:10 UTC.
Public submission 364647, private submission 364648, run 31388 completed.
Public leaderboard score **0.1947902971** (submission display **0.1948**),
rank 128 at observation. On October 4, MAIN daily quota **2/3 remained** at
observation, with no authorization for either remaining attempt. A daily reset
does not authorize another MAIN. DEBUG v2 remains a separate practice score 0.3331.
Server runtime, detailed logs, and private score were not exposed.

The exact release is read-only under `/data/terradelta/releases/v2-main-01/`.
All 33 files were hash-verified after copying. Its ZIP SHA-256 is
`bb437aa1d3dcbf7a1ab053281b37bdbb8804faaa6cc91458250a9fb0c7ef8f43`;
step250 SHA-256 is
`ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372`.
Source branch `v2-siamese-20261004` remains at
`c9b1c136ee7af9cf68b23e2f63f60b981da5724c`.
The release was frozen while external scoring continued. Its original receipt
is unchanged; the completed MAIN result is the read-only sibling
`/data/terradelta/releases/v2-main-01-result.json` and `v2-main-result.json` here.

## Training contract fixed before GPU creation

Branch `v2.1-season-aware-20261004` was created only after MAIN registration and
v2 freeze. Use exactly the shared Siamese ResNet18, alignment/fusion, independent
pixel and presence heads. The stable photometric consistency fallback avoids an
unvalidated adversarial nuisance head and keeps the same inference architecture
and checkpoint loader. No pretrained weights or encoder download is needed.

Each training pair receives shared geometry, PRE-only registration bounded to the existing v2 range and
independent bounded seasonal/vegetation tint, gamma, shadow, haze, local light,
contrast and blur. A second appearance view keeps exactly the same geometry and
labels. Both views receive the supervised loss; probability MSE consistency is
added for valid pixels and valid presence labels. Empty targets remain empty.
Batch-normalization running statistics are frozen to avoid updating them twice
per pair and to retain v2's inference normalization.

One comparison: lambda **0.05 versus 0.10**, each initialized from frozen step250
with the same seed/data draws. Checkpoints at **25, 50, 100, 150 additional steps**;
no 250+ fine-tuning. Encoder LR 5e-6, remaining parameters 5e-5. Batch 8 keeps
2 building, 2 tree, 1 both and 3 negative samples. Negative draws have a **3:1
real:synthetic category ratio**, rather than relying on the 54:700 pool sizes.
This actively oversamples individual real negatives while preserving positive
category balance. Each checkpoint records actual sampled category counts.

## Actual data audit

The original v2 trainer used only 2,200 synthetic rows. This new manifest has
2,261 rows: 705 building, 702 tree, 100 both, 700 same-scene synthetic negatives,
and 54 reviewed real negatives (NC 22 / MD 32). Real positive rows retain their
partial review masks. Original manifests/images/masks are not modified.

All 54 real negatives have actual zero masks, full validity, accepted full-tile
review records and verified provenance/rights. The WA real validation contains
3 building, 2 tree and 17 no-change pairs; WA synthetic stress remains 470 pairs,
including 200 negatives. Training/validation region, source ID, path and exact
PRE/POST image-hash overlap are all zero. See `v21-data-audit.json` for hashes.
These are existing AI visual reviews, not independent human annotations; 17
real negatives are a small diagnostic population, not a generalization proof.

## Selection and submission gates

Evaluate actual exported polygons with the same frozen v2 per-class presence /
pixel thresholds and reference geometry (area30/20, simplify0.5, ndigits2).
Re-evaluate frozen v2 on the same GPU and validation populations first.
An eligible candidate must reduce the real no-change FP count, preserve both
class positive TP counts on both sets, retain at least 95% of each class shape
score, avoid increasing synthetic no-change FP, and not reduce either overall
score. Rank eligible candidates by real negative FP, real score, stress score,
then shorter training. Training progress alone is never promotion evidence.

At most one v2.1 DEBUG is allowed after a candidate passes local gates and the
extracted offline CPU cleanroom. Compare its practice score to 0.3331. No MAIN
submission is authorized for v2.1; retain frozen v2 if improvement is not proven.

## Completed result

Both 150-step conditions completed successfully in 194.92 seconds total, including the frozen v2 reference evaluation. All eight checkpoints failed the promotion gates. No v2.1 DEBUG, ZIP submission or further MAIN was attempted.

| Model / lambda | Extra steps | Real no-change FP | Real building/tree TP | Real score | Synthetic no-change FP | Synthetic building/tree TP | Synthetic score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen v2, same GPU | 0 | 17/17 | 3/3, 2/2 | 0.508952 | 42/200 | 130/150, 104/150 | 0.635719 |
| 0.05 | 25 | 15/17 | 3/3, 1/2 | 0.411790 | 0/200 | 31/150, 58/150 | 0.368158 |
| 0.05 | 50 | 16/17 | 3/3, 1/2 | 0.432133 | 0/200 | 65/150, 70/150 | 0.455806 |
| 0.05 | 100 | 12/17 | 3/3, 1/2 | 0.335362 | 0/200 | 94/150, 85/150 | 0.578537 |
| 0.05 | 150 | 15/17 | 3/3, 1/2 | 0.338407 | 0/200 | 106/150, 84/150 | 0.596301 |
| 0.10 | 25 | 15/17 | 3/3, 1/2 | 0.411034 | 0/200 | 31/150, 58/150 | 0.368157 |
| 0.10 | 50 | 16/17 | 3/3, 1/2 | 0.432375 | 0/200 | 65/150, 70/150 | 0.456320 |
| 0.10 | 100 | 12/17 | 3/3, 1/2 | 0.334175 | 0/200 | 94/150, 84/150 | 0.574750 |
| 0.10 | 150 | 15/17 | 3/3, 1/2 | 0.339757 | 0/200 | 106/150, 86/150 | 0.598616 |

The lowest real FP candidate is lambda .05 / step100, but its real building shape score drops from .882652 to .135413 and real tree recall from 2/2 to 1/2. Its real score is .335362 and synthetic score .578537. It is a diagnostic checkpoint only, not a promoted release. The best real-score candidate (.10/step50, .432375) also loses tree recall and synthetic recall. Reducing false positives at this cost does not establish improvement.

The same-GPU frozen reference differs from the previous CPU score by about 1.14e-5 on real and 5.02e-6 on synthetic shapes, with identical presence confusion and no-change counts. Every comparison above uses the new same-GPU reference and unchanged frozen thresholds.

Each arm actually sampled 300 building + 300 tree + 150 both + 336 real-negative + 114 synthetic-negative pairs (1,200 draws). The real/synthetic negative ratio is 2.947:1. The two arms used the same draws.

Local verification: 608 passed / 1 CUDA test skipped; repository Ruff passed. The remote CUDA environment ran 115 tests / 1 skipped because the shared test fixture masks CUDA. The completed training and eight evaluations provide actual A10G CUDA execution proof. Project extras train+geo were installed on the GPU only; geo is required by the existing training package imports.

All 33 frozen v2 release files were hash-verified again after training. All training weights, logs and original inputs remain on the retained 30 GiB disk. `v21-comparison.json`, `v21-data-audit.json`, `v21-source-identity.json` and `v21-training-terminal.json` provide the small reproducibility records. Final AWS stopped states and volume restoration are recorded in `v21-aws-final.json`.

**V2.1 NOT PROVEN — KEEP FROZEN V2**
