# Micro pilot v2 — 2026-10-04

TerraDelta positive-GT refinement progressed the retained EC2 pilot from
`NOT READY` to **READY WITH WARNINGS** for a very small, exploratory short
fine-tuning experiment. No model training, optimizer update, GPU operation or
AIFactory submission was performed in this step.

## Dataset

The dataset remains on the retained TerraDelta EBS under
`/data/terradelta/processed/micro-pilot-v2`; imagery and masks are not committed
to Git.

- Total reviewed samples: 83
- Full-tile reviewed no-change samples: 71
- Positive samples: 12
  - new_building: 8
  - tree_removal: 4
- Train: 61 samples, NC Wake + MD Laurel
  - 5 building positives
  - 2 tree-removal positives
- Validation: 22 samples, WA Olympia only
  - 3 building positives
  - 2 tree-removal positives
- Spatial leakage candidates: 0
- Temporal leakage candidates: 0
- Real coordinate evidence: 83/83

Seven positive training samples use conservative positive-only review masks.
Pixels outside each reviewed positive region are ignored rather than treated as
background. Validation contains only full-tile masks.

The labels were visually reviewed by ChatGPT from the native 256x256 PRE/POST
imagery and available candidate evidence. They are not represented as an
independent professional/human annotation benchmark.

## Audit

Audit output:
`/data/terradelta/audit/micro-pilot-v2`

Result: **READY WITH WARNINGS**, with zero blockers.

Warnings are limited to:
- seven partial positive-only training labels, intentionally excluded from validation;
- two large WA development tiles with unreliable phase-correlation estimates because
  the scene itself changed substantially.

Class distribution among 83 samples:

- background-only: 71
- new_building-only: 8
- tree_removal-only: 4
- both classes: 0

Valid-pixel ratios are approximately 96.30% background, 3.01% new building and
0.685% tree removal. This is a deliberately small pilot, not a balanced final
training set.

## Official baseline reference

The organizer baseline ZIP was downloaded directly by the EC2 builder, not via the
user's hotspot. The unchanged checkpoint reports:

- UNet / ResNet18
- 6-channel PRE+POST RGB
- 3 classes
- checkpoint steps = 100

CPU-only inference on the 22-sample WA validation split produced the following
**approximate local metric**:

- overall: 0.470028
- new_building: 0.493284
- tree_removal: 0.446773
- no-change false-positive rate: 70.59%

Presence confusion:

- new_building: TP 3 / FP 13 / FN 0 / TN 6
- tree_removal: TP 2 / FP 3 / FN 0 / TN 17

This local score is for A/B comparison only and must not be interpreted as a
leaderboard prediction. The small validation set and approximate evaluator make
absolute score conclusions unsafe. The high no-change false-positive rate does,
however, justify prioritizing conservative thresholds and very short fine-tuning.

## Next experiment

If training is explicitly approved later, the first useful experiment is a short
checkpoint sweep from the official baseline:

`25 / 50 / 75 / 100` optimizer steps first, while tracking no-change false
positives, then `150 / 250` only if the early curve supports it.

Current data quality is sufficient for an exploratory short run, but not for a
strong final model. More independently reviewed positives—especially tree-removal
examples and a second geographic validation region—remain valuable.
