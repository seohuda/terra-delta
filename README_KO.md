# TerraDelta

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Archived](https://img.shields.io/badge/Status-Archived-inactive.svg)](#project-status)
[![Final Rank](https://img.shields.io/badge/Final%20Rank-86%20%2F%20136-orange.svg)](#final-public-result)
[![Best Public Score](https://img.shields.io/badge/Best%20Public%20Score-0.3567265623-brightgreen.svg)](#final-public-result)

2026 국립공원 위성 모니터링 AI 챌린지 주제 3: 국립공원 내 시설물 변화 탐지를 위해 개발한 고해상도 위성·항공 영상 기반 시계열 변화 탐지 프로젝트입니다.

영문 README: [README.md](README.md)

대회 페이지: https://aifactory.space/ko/competitions/9306

프로젝트 상태: 2026-10-06 대회 종료. 현재 저장소는 기술 기록과 재현 참고용 아카이브로 유지합니다.

## 최종 공개 결과

| 항목 | 결과 |
| --- | ---: |
| 최종 공개 순위 | 86 / 136팀 |
| 최고 공개 점수 | 0.3567265623 |
| 총 제출 횟수 | 14회 |
| 최고 공개 모델 계열 | V3 Satlas |
| 대회 종료일 | 2026-10-06 |

위 값은 대회 종료 후 AIFactory 리더보드 화면에서 확인한 최종 공개 결과입니다.

`docs/v31-final-result.json`에는 V3.1 제출 과정에서 기록된 별도의 private score 메타데이터가 남아 있습니다. 종료 화면의 공개 리더보드 값과 다르므로 저장소 대표 성과는 위 공개 결과를 기준으로 정리했습니다.

## 점수 변화

| 버전 | 핵심 변경 | Public 점수 |
| --- | --- | ---: |
| V2.0 | Siamese ResNet18 | 0.1947902971 |
| V2.2 | Season-aware verifier | 0.2045954238 |
| V2.3 | Geometric stability gate | 0.2484213920 |
| V2.3.1 | Object-evidence classifier | 0.2899243555 |
| V3 | Satlas Swin-v2 + AIHub 71363 | 0.3567265623 |

가장 큰 성능 향상은 ResNet18 기반 후처리를 계속 복잡하게 만드는 대신, 항공·위성 도메인에 맞는 Satlas 사전학습 백본으로 표현 자체를 바꿨을 때 발생했습니다.

## 최종 공개 모델 구조

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

핵심 구성:

- PRE/POST에 동일한 Satlas Swin-v2 백본 사용
- 단순 temporal max pooling 대신 방향성을 보존하는 시계열 차이 특징 사용
- `new_building`, `tree_removal`을 독립적으로 예측
- 클래스별 presence gate와 polygon threshold 적용
- AIHub 71363 SkySat 시계열 건물 변화 샘플 활용
- tree annotation이 없는 AIHub 샘플에서는 tree loss를 마스킹
- 실제 목표 도메인과 가까운 no-change negative를 활용해 오탐 억제
- 인터넷 없이 A10G 환경에서 실행되는 제출 패키지 구성

## 로컬 검증 결과

승격한 V3 S2 step 800 기준:

| 검증 세트 | Score | Building | Tree | No-change FP |
| --- | ---: | ---: | ---: | ---: |
| Real Legacy | 0.631827 | 3 / 3 | 2 / 2 | 7 / 17 |
| Stress | 0.775560 | 110 TP | 119 TP | 0 / 200 |

로컬 검증 점수와 실제 Public 점수 사이에 큰 차이가 있었습니다. 이 차이는 프로젝트의 중요한 실패·학습 결과로 그대로 남겼습니다.

## 저장소 구조

```text
configs/              실험 및 추론 설정
docs/                 실험, 제출, 감사, 결과 기록
scripts/              데이터 준비, 학습, 평가, 패키징
src/terradelta/       모델, 데이터, 추론, 후처리, 평가 코드
submission/template/  오프라인 제출 노트북 템플릿
tests/                회귀 및 추론 테스트
notebooks/            소규모 분석 노트북
outputs/              가벼운 메타데이터만 추적
tasks/lessons.md      개발 과정에서 정리한 회고
```

원본 위성 영상, 대용량 체크포인트, 캐시, 제출 ZIP은 저장소에 포함하지 않습니다.

## 데이터 및 라이선스

개발 과정에서 사용하거나 검토한 주요 데이터:

- 대회 공식 데이터
- AIHub 71363 고해상도 시계열 위성 영상
- NAIP 항공 영상
- Hansen Global Forest Change: 이벤트 후보 탐색 용도
- Microsoft GlobalML Building Footprints: 정적 건물 참고 용도

정적 건물 footprint나 저해상도 산림 변화 이벤트를 시간 변화 segmentation GT로 직접 사용하지 않았습니다.

세부 출처와 라이선스 기록은 다음 파일에 있습니다.

- `LICENSE_DATA.md`
- `THIRD_PARTY_NOTICES.md`

## 설치

Python 3.10 이상을 권장합니다.

```bash
git clone https://github.com/seohuda/terra-delta.git
cd terra-delta

python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,train,geo]
pytest -q
```

정확한 학습 재현에는 저장소에 배포하지 않는 외부 데이터와 사전학습 가중치가 추가로 필요합니다.

V3 관련 주요 진입점:

```bash
python scripts/prepare_v3_datasets.py --help
python scripts/train_satlas_v3.py --help
python scripts/evaluate_satlas_v3.py --help
python scripts/evaluate_ensemble_v3.py --help
python scripts/package_v3_satlas.py --help
```

## 제출 패키지

대회 제출 패키지는 인터넷 연결 없이 동작하도록 구성했습니다.

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

출력 형식:

```text
id,new_building,tree_removal
```

V3 제출본은 cleanroom 환경에서 반복 실행 결과가 동일한지 검증한 뒤 제출했습니다.

## 개발 타임라인

- 2026-10-04: baseline 재현, 데이터 감사, real-data pilot
- 2026-10-04 ~ 2026-10-05: V2 ~ V2.3.2 실험
- 2026-10-06: Satlas V3 + AIHub 71363 학습 및 최종 제출
- 2026-10-06: 대회 종료
- 최종 공개 리더보드: 86 / 136, 0.3567265623

## 프로젝트에서 얻은 점

- 후처리 규칙을 계속 추가하는 것보다 도메인에 맞는 pretrained representation이 훨씬 큰 영향을 줬습니다.
- 짧고 목적이 명확한 학습이 무작정 오래 학습하는 것보다 효율적이었습니다.
- 실제 목표 도메인과 가까운 negative가 false positive 억제에 중요했습니다.
- `tree_removal` 실제 양성 데이터 부족이 가장 큰 데이터 병목이었습니다.
- 좁은 local validation만으로 hidden test 일반화를 판단하기 어려웠습니다.
- 제출 직전에는 실제 A10G 환경에서 메모리와 실행 시간을 확인하는 것이 중요했습니다.

## 라이선스

별도 표기가 없는 소스 코드와 문서는 MIT License를 따릅니다.

외부 데이터와 third-party asset은 각 원본 라이선스를 그대로 따릅니다. 대회 원본 영상, AIHub 원본 데이터, 학습된 모델 가중치, 제출 ZIP은 이 저장소에서 재배포하지 않습니다.

TerraDelta는 더 이상 대회용으로 개발하지 않으며, 현재 저장소는 실험 기록과 참고 구현을 보존하기 위한 아카이브입니다.
