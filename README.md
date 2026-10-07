# TerraDelta

[한국어 README](README_KO.md)

Satellite change detection for national-park monitoring with paired high-resolution PRE/POST RGB imagery.

TerraDelta was developed for the 2026 National Park Satellite Monitoring AI Challenge, Track 3: 국립공원 내 시설물 변화 탐지.

Competition: https://aifactory.space/ko/competitions/9306

Project status: concluded on 2026-10-06. The repository is preserved as a technical archive and reference implementation.

## Final public result

| Item | Result |
| --- | ---: |
| Final public rank | 86 / 136 teams |
| Best public score | 0.3567265623 |
| Total submissions | 14 |
| Best public model family | V3 Satlas |
| Competition end date | 2026-10-06 |

The final public leaderboard values above are taken from the AIFactory competition UI captured after the competition ended.

A separate historical submission receipt in `docs/v31-final-result.json` contains a later V3.1 private-score record. It is preserved as submission metadata, but it is not used as the headline final public result because it does not match the final public leaderboard screen.

## Public score progression

| Version | Main idea | Public score |
| --- | --- | ---: |
| V2.0 | Siamese ResNet18 | 0.1947902971 |
| V2.2 | Season-aware verifier | 0.2045954238 |
| V2.3 | Geometric stability gates | 0.2484213920 |
| V2.3.1 | Object-evidence classifier | 0.2899243555 |
| V3 | Satlas Swin-v2 + AIHub 71363 | 0.3567265623 |

The largest gain came from replacing the original ResNet18-centered stack with a domain-matched aerial foundation backbone.

## Final public model architecture

```text
PRE RGB ──┐
          ├─ shared Satlas Swin-v2 encoder
POST RGB ─┘
          ↓
multi-scale temporal fusion
[PRE, POST, |POST-PRE|, POST-PRE]
          ↓
FPN / decoder
          ↓
building + tree segmentation/presence heads
          ↓
pixel-coordinate polygons
```

Key characteristics:

- shared Satlas Swin-v2 backbone for both timestamps
- directional temporal features rather than symmetric temporal max pooling
- independent `new_building` and `tree_removal` outputs
- class-specific presence gates and polygon thresholds
- AIHub 71363 SkySat temporal building-change examples
- tree supervision masked on AIHub samples without tree labels
- target-like no-change negatives for false-positive suppression
- fully offline A10G submission packaging

## Promoted V3 local validation

| Validation set | Score | Building | Tree | No-change FP |
| --- | ---: | ---: | ---: | ---: |
| Real Legacy | 0.631827 | 3 / 3 | 2 / 2 | 7 / 17 |
| Stress | 0.775560 | 110 TP | 119 TP | 0 / 200 |

The public leaderboard score was substantially lower than local validation. That local-to-public gap is retained as an important result of the project rather than hidden.

## Repository structure

```text
configs/              experiment and inference configuration
docs/                 experiment, submission, audit, and result records
scripts/              data preparation, training, evaluation, packaging
src/terradelta/       models, data, inference, post-processing, metrics
submission/template/  offline competition notebook template
tests/                regression and inference tests
notebooks/            small analysis notebooks
outputs/              lightweight tracked metadata only
tasks/lessons.md      retrospective engineering notes
```

Large imagery, model checkpoints, caches, and generated submission ZIP files are intentionally not distributed in this repository.

## Data and provenance

Sources used or evaluated during development included:

- official competition data
- AIHub 71363 high-resolution temporal imagery
- NAIP aerial imagery
- Hansen Global Forest Change for event discovery only
- Microsoft GlobalML Building Footprints as static reference only

Static footprints and low-resolution event products were not treated as direct temporal segmentation ground truth without supporting temporal evidence.

See `LICENSE_DATA.md` and `THIRD_PARTY_NOTICES.md` for provenance and license notes.

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/seohuda/terra-delta.git
cd terra-delta

python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,train,geo]
pytest -q
```

Model training and exact competition reproduction additionally require external data and pretrained weights that are not distributed here.

Useful V3 entry points:

```bash
python scripts/prepare_v3_datasets.py --help
python scripts/train_satlas_v3.py --help
python scripts/evaluate_satlas_v3.py --help
python scripts/evaluate_ensemble_v3.py --help
python scripts/package_v3_satlas.py --help
```

## Submission format

The competition package was self-contained and designed for offline execution.

```text
predict.ipynb
requirements.txt
LICENSE
NOTICE
assets/
  model/model.pt
  config.yaml
  code/
```

Prediction output:

```text
id,new_building,tree_removal
```

The promoted V3 package was verified with deterministic cleanroom execution before submission.

## Timeline

- 2026-10-04: baseline reproduction, data audit, and real-data pilot
- 2026-10-04 to 2026-10-05: V2 through V2.3.2 experiments
- 2026-10-06: Satlas V3 + AIHub 71363 and final submissions
- 2026-10-06: competition ended
- final public leaderboard: 86 / 136, score 0.3567265623

## Lessons

- domain-matched pretrained representations mattered more than stacking additional shallow heuristics
- short, carefully selected training runs were often more useful than training longer
- target-like negatives were essential for false-positive control
- tree-removal supervision remained the largest data bottleneck
- local validation needed broader geographic and sensor diversity
- actual A10G deployment constraints needed to be checked before submission

## License

Source code and repository documentation are released under the MIT License unless a file states otherwise.

Third-party assets and data retain their original terms. Competition imagery, AIHub raw data, trained model weights, and submission archives are not redistributed here.

TerraDelta is no longer under active competition development. The repository is retained as an experiment archive and reference implementation.
