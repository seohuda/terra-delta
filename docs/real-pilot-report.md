# TerraDelta 실제 CPU pilot 결과 — 2026-10-04

**NOT READY.** EC2에서 실제 시계열 후보 644개를 구축했고, 전체 타일을
검토한 무변화 71개를 승인했습니다. 신규 건물과 벌채는 실제 영상에
보이지만 정답 경계가 승인되지 않았습니다. 목표한 200–500개, 또는 품질이
검증된 100–200개의 균형 잡힌 GT pilot은 **확보하지 못했습니다**.
모델 학습, optimizer update, GPU 생성/시작, 대회 제출은 하지 않았습니다.

검토자는 Codex의 AI 영상 검토입니다. 독립적인 사람의 라벨 검수가 완료된
데이터셋으로 표현하지 않습니다. HIGH는 해당 타일에서 관찰한 두 클래스의
부재에 대한 판단이며, 모델 탐지 성능이나 양성 경계 품질을 뜻하지 않습니다.
정답 파일의 단순 무결성 검사와 클래스·검증 데이터 준비 여부도 구분했습니다.

[기계 판독 요약](real-pilot-summary.json)에 실제 audit, AWS API 최종 상태,
원본 파일 보존 확인과 모든 수치를 담았습니다. 이미지와 원본은 Git에 없습니다.

## 요청한 37개 항목

| # | 항목 | 확인 결과 |
| --- | --- | --- |
| 1 | TerraDelta instance ID | `i-0766a472ecb5bcf88`, Name=terradelta-data-builder, Project=terra-delta, Purpose=dataset-builder |
| 2 | Instance type | `t3.medium`, 2 vCPU / 4 GiB, CPU credit **standard**, GPU 없음 |
| 3 | Region | `ap-northeast-2`, AZ `ap-northeast-2c` |
| 4 | EBS | 암호화 30 GB gp3, `vol-0070845086ec08190`, 3000 IOPS / 125 MB/s, DeleteOnTermination=false. 40 GB 증설 불필요 |
| 5 | 최종 EC2 상태 | **stopped**, 04:13:00 UTC API 검증. STOP 요청 04:12:11 UTC, 공인 IPv4 없음, 전용 SG inbound 없음 |
| 6 | 실행 시간·비용 | 03:00:17 UTC 생성 → 04:12:11 STOP 요청: 71분 54초. stopped 검증까지 약 72분 44초. CPU $0.052/h + IPv4 $0.005/h로 계산한 약 **$0.0683–0.0691**, EBS 별도. 실제 청구 확정액 아님 |
| 7 | GPU 사용 | 생성·시작·사용 없음. CPU PyTorch 환경 약 1.5 GB |
| 8 | 실제 pair 수 | 실제 temporal 후보 **644**; 최종 승인 GT pair **71**. 두 수치는 서로 다른 단계이며 합산하지 않음. mock 4개는 PC 테스트 전용, 실제 pilot에 0개 |
| 9 | No-change | 승인 **71**, 각 샘플의 mask.png는 전체 0. 계절·밝기·그림자·차량·기존 지붕 색 변화가 포함된 hard negatives |
| 10 | New building | 승인 **0**. 영상에서 신규 창고가 확인되지만 전체 타일 정답 경계는 미승인 |
| 11 | Tree removal | 승인 **0**. Hansen·NIR 기반 후보와 실제 clearing은 있으나 경계/그림자 누락이 있어 미승인 |
| 12 | Regions | WA Olympia: 후보 180 / 승인 17; NC Wake: 240 / 22; MD Laurel: 224 / 32. 서로 다른 세 지리·source raster pair |
| 13 | 촬영 시점 | WA 2019-10-10 → 2023-08-14; NC 2018-08-26 → 2022-07-20; MD 2018-10-19 → 2021-06-17. EC2 STAC 탐색·실제 헤더로 선택; 연도 추측 아님 |
| 14 | 평균 resolution | **0.6 m/px**, 모든 최종 타일 256×256 RGB, PRE 3ch + POST 3ch. 원본의 4번째 NIR 밴드는 후보 검토용으로 별도 보관 |
| 15 | Raw 크기 | 11개 실제 source payload 합계 **3,014,013,440 bytes**. Microsoft index·작은 license metadata는 이 수치에 별도. 전국 데이터 전체 수집 아님 |
| 16 | Processed 크기 | 승인 pilot 폴더 **17,839,809 bytes**, audit가 참조한 영상·mask·valid-mask **16,496,947 bytes**. 644개 후보 폴더 **220,306,240 bytes** 별도. 원본·후보·검토 metadata 모두 EBS에 보존 |
| 17 | 거절 수 | 명시적 LOW/reject **3**. 추가 검토 75개와 미검토 495개를 거절 완료 수로 세지 않음 |
| 18 | Confidence | HIGH **71** / MEDIUM **75** / LOW **3** / UNREVIEWED **495**. HIGH만 승인 manifest에 포함. MEDIUM에는 양성 후보 및 무변화 여부가 불확실한 타일 모두 포함 |
| 19 | Alignment | 승인 71개 모두 reliable: mean **1.0048**, median **1.0858**, p90 **1.7363**, p95 **1.8174**, p99 **1.9083**, max **1.9384 px**. dx/dy는 EBS audit/alignment.csv에 기록. 영상 좌표 수정 없음 |
| 20 | Spatial leakage | 승인 train/val 간 후보 **0**; 실좌표 71/71, source raster 연결·인접 footprint grouping 검사, decoded image 중복 0 |
| 21 | Temporal leakage | 승인 train/val 간 후보 **0**; 촬영 연도와 source raster의 시간 역할 교차 검사. 임의 타일 random split 없음 |
| 22 | Suspicious samples | 승인 subset **0**; RGB/크기/손상/잘못된 mask/중복 오류 0. 전체 후보 RGB 손상·크기 오류 0, 원래 candidate manifest에는 GT 누락 644개(승인 export와 구분) |
| 23 | Geographic split | train **39** = WA 17 + NC 22; val **32** = MD 32. 세 연결 geography를 통째로 분할. 현재 둘 다 양성 GT가 없어 탐지 검증 불가 |
| 24 | 실제 사용 source | NAIP, Hansen GFC 2024 v1.12, Microsoft GlobalML 2026-07-24. FEMA/Google/LEVIR/AIHub는 이 pilot 입력으로 사용하지 않음 |
| 25 | License별 sample 수 | 최종 승인 pair **71: NAIP public-domain 근거 + 관찰한 부재의 0 mask**. Hansen CC BY 4.0 / GlobalML CDLA-Permissive-2.0은 후보 근거로만 사용, 각각 최종 GT 0개. raw 파일 수는 아래 표 |
| 26 | 미확인 license | 사용한 세 source의 공식 근거 보존. FEMA는 CC BY 4.0 텍스트와 BY 3.0 badge 불일치 때문에 제외. candidate manifest의 unknown은 미승인 라벨까지 포함한 보수적 gate이며 승인된 데이터의 미확인 imagery license를 뜻하지 않음 |
| 27 | Readiness | **NOT READY** |
| 28 | 이유 | 승인된 new_building / tree_removal mask가 모두 없고, balanced GT 수량 목표도 미달. audit에 두 missing-target-class blocker를 명시; negatives-only 데이터를 READY로 오판하지 않도록 코드 수정 |
| 29 | 학습 금지 준수 | train.py 실행, fine-tuning, optimizer.step, GPU, submission 모두 없음. baseline inference도 이번 pilot에 실행하지 않음 |
| 30 | PC 대용량 전송 | raw dataset, 환경, weights를 PC에 다운로드/중계하지 않음. PC의 preview + 작은 summary JSON 합계 **11,393,132 bytes**, 20 MB 이하. SSH/Git/log/protocol traffic 별도 |
| 31 | 실제 전송 경로 | Provider → 전용 EC2. raw 파일 SHA-256 재검증 11/11 통과. S3 신규 생성·기존 bucket 재사용 없음 |
| 32 | Git commit | 실제 EC2 처리 코드 `a9fab650b29f7781b2d5a58b7ce2b8ecebdf0113`. 이 보고서를 포함한 후속 documentation commit의 최종 SHA는 전달 응답/Git HEAD에 기록(자기 commit SHA를 파일 안에 순환 기록하지 않음) |
| 33 | Remote/local SHA | 코드 commit은 GitHub main에 동일 Git objects로 게시·검증. 최종 문서 commit도 게시 후 GitHub/local SHA 일치 확인; force update 없음 |
| 34 | Working tree | 최종 전달 시 small code/docs/metadata만 commit, clean 여부를 다시 확인. 실제 dataset·previews·원본·checkpoint는 .gitignore로 제외 |
| 35 | Redstar 상태 | `i-0426ce1098082fee2`, c8i.xlarge, **stopped** 유지. 기존 bot-server `i-0db8b62f43d95f150`는 running 유지, 두 리소스 모두 변경 없음 |
| 36 | Redstar 잔여 비용 | `vol-0fd71ecf4d0529fc7`: 80 GB gp3, 월 nominal **$7.296**. DeleteOnTermination=true, 데이터 미검사. 명확한 Redstar EIP/추가 orphan EBS/owned snapshot/NAT/LB/TG는 scoped 조사에서 없음 |
| 37 | Redstar 삭제/보존 | **삭제·수정 없음.** EBS는 REVIEW, 전용 ENI/SG/VPC/subnet/IGW/route table/key는 데이터 보존·의존성 재검사 후 삭제 후보. 다른 서비스 EIP·공유 리소스 KEEP. [상세 inventory와 삭제 계획](redstar-inventory.md) |

실제 Cost Explorer 귀속액은 알 수 없습니다. Project cost-allocation tag가
Inactive였으므로 빈 조회를 “실제 비용 0”으로 해석하지 않았습니다. Seoul
gp3 $0.0912/GB-month 기준 builder EBS는 **$2.736/월**, Redstar와 합하면
nominal **$10.032/월**입니다. 세금·할인·크레딧 전의 보관 비용이며 EC2를
중지해도 남습니다. 중지 후 builder의 CPU·동적 공인 IPv4 실행 비용은 끝납니다.

## 후보 전체와 승인 subset의 구분

644개 원본 candidate 파일은 변경하지 않았습니다. 승인 데이터는 새
`/data/terradelta/processed/pilot`에 별도 export했습니다. 따라서 원래 후보
audit의 “missing GT 644”와 승인 export의 “정상 mask 71”은 모순되지 않습니다.
승인 subset에 없는 **573**개는 review_queue.csv에 남습니다.

전체 후보의 phase-correlation reliable 560 / unreliable 84. Reliable subset의
mean 3.5028, median 1.3393, p90 10.3700, p95 15.9726, p99 33.4982,
max 57.0280 px; >4 px 후보 **101**개입니다. 개발로 영상 내용 자체가 바뀌면
phase 추정이 잘못될 수도 있으므로 이를 실측 지리 오차로 단정하지 않았습니다.
큰 값과 불안정한 값은 미승인 상태이며, unchanged control feature로 재검토해야
합니다. 승인 71개에는 큰 shift나 unreliable 결과가 없습니다.

실제 신규 창고를 static footprint가 누락하는 사례와 기존 창고의 지붕 색
변화를 신규 건물로 제안하는 사례를 모두 확인했습니다. Hansen 30m event와
NIR 조건만으로 만든 벌채 제안은 shadowed crown을 누락하고 구멍이 생깁니다.
이 오류를 0.6m GT로 승인해 수량을 채우지 않았습니다. 149개 타일의 명시적
decision은 EBS `metadata/pilot-review-decisions.json`과 승인 export 안에 보존됩니다.

## Actually used sources

| Source | Provider / 공식 URL | License·상업 사용·파생·재배포 | Attribution / 실제 입력 |
| --- | --- | --- | --- |
| NAIP | USDA FPAC/USGS; [USDA catalog](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) | Catalog public-domain 지정 근거에 따라 사용·파생·재배포. 임의 basemap 권한으로 확대하지 않음 | USDA/product/item/date/unsigned asset URL 보존; raw raster 6개, candidate pair 644, 승인 pair 71 |
| Hansen GFC 2024 v1.12 | Hansen/UMD/Google/USGS/NASA; [versioned provider](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): 상업 사용·파생·재배포 가능, credit/license/change notice 유지 | Source: Hansen/UMD/Google/USGS/NASA; Hansen et al. (2013), Science 342:850–853, [DOI](https://doi.org/10.1126/science.1244693); raw lossyear raster 2개, AOI window reproject/candidate mining만 수행, GT 0 |
| Microsoft GlobalML 2026-07-24 | Microsoft; [GlobalML license](https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/LICENSE) | [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/): use/derivatives, 공유 data에 agreement 포함 | Microsoft/product/version/source 유지; quadkeys 021230201, 032012002, 032010032 gzip 3개를 AOI stream/filter/rasterize. Static reference만, GT 0 |

Agreement/provider pages와 정확한 raw URL·SHA-256·크기를 EBS metadata에
보존했습니다. SAS credential은 메모리에서만 사용했고 persisted URL은 unsigned입니다.
FEMA geometry는 받아 쓰지 않았습니다. Hansen과 Microsoft는 위 PRE/POST 검토에
쓰인 후보 자료이며 최종 배경 GT에 외부 label shape를 넣지 않았습니다.

## 보존·검증·재현

보존 확인 04:10:09 UTC: 전체 `/data/terradelta` file bytes
3,267,034,801(확인 metadata 자체 쓰기 전 snapshot), EBS filesystem used
7,713,370,112 / free 22,353,629,184 bytes. Pilot 데이터 자체는 10GB 목표보다
작고 30GB EBS로 충분했습니다. 11개 raw SHA-256/size 재검증 뒤 sync했고,
DeleteOnTermination=false를 API로 확인한 뒤 STOP했습니다. 이것은 EBS 보존이며
별도 disaster-recovery backup이 아닙니다. Snapshot/S3를 억지로 만들지 않았고
terminate도 하지 않았습니다. **해당 EBS를 삭제하면 유일한 전체 데이터 사본을 잃습니다.**

SSM/SSH 터널과 임시 read-only GitHub deploy key(ID 165311355)를 정리했습니다.
전용 SSM IAM role/profile과 SSH key·SG는 재접속용으로 보존합니다. SG inbound는
비어 있습니다. 기존 default VPC/subnet은 읽기 전용으로 선택했고, 다른 프로젝트의
리소스·bucket을 변경하지 않았습니다. Redstar 데이터도 그대로입니다.

이 단계의 data/external/audit/review 테스트 **159 passed**, Ruff 및 diff check
통과. 실제 EC2에서 CPU setup, raw decode, 644개 후보 생성, 71개 decision
export, 승인 audit와 전체 후보 audit를 실행했습니다. 기존 audit 디렉터리를
덮어쓰지 않았습니다. 준비 코드 검증과 학습 성능 검증을 혼동하지 않습니다.

EC2 데이터 위치:

```text
/data/terradelta/raw/{naip,hansen,microsoft}/
/data/terradelta/metadata/{acquisition-plan,download-report,pilot-review-decisions,preservation-check}.json
/data/terradelta/review/candidates/       # 644 pairs; evidence/proposals, GT 아님
/data/terradelta/processed/pilot/         # 71 pairs + manifest/train/val/review_queue
/data/terradelta/audit/{pilot,candidates}/
/data/terradelta/previews/pilot_result.jpg
```

재검토 decision을 명시한 뒤 사용하는 offline export/audit 명령은 다음과 같습니다.
새 output 이름을 사용하며, instance 재시작이나 학습을 수행하는 명령이 아닙니다.

```sh
python scripts/pilot_review.py \
  --candidates /data/terradelta/review/candidates/candidates.csv \
  --decisions /path/to/explicit-reviewed-decisions.json \
  --output /data/terradelta/processed/new-reviewed-pilot
python scripts/audit_data.py \
  --manifest /data/terradelta/processed/new-reviewed-pilot/manifest.csv \
  --train-manifest /data/terradelta/processed/new-reviewed-pilot/train.csv \
  --val-manifest /data/terradelta/processed/new-reviewed-pilot/val.csv \
  --scan-root /data/terradelta --output /data/terradelta/audit/new-reviewed-pilot
```

마지막 category montage는 PRE | POST | GT | overlay입니다. Building/tree/rejected
행의 GT는 회색 **PENDING / NOT APPROVED**로 표시하고 overlay도 **PROPOSAL ONLY /
NOT GT**로 구분했습니다. No-change/hard-negative 행만 실제 승인된 0 GT입니다.
자료는 EBS와 PC의 `outputs/pilot-previews/pilot_result.jpg`에 있으며 Git에는 없습니다.

**25 / 50 / 75 / 100-step 학습은 현재 가치가 없습니다.** 배경만 학습하면 두
양성 클래스를 학습하거나 평가할 수 없습니다. 다음 단계는 실제 양성 타일의
고해상도 경계 refinement와 독립 검수, 양성 validation 샘플 확보입니다. 이후
class-balanced audit를 통과해야 짧은 실험을 비교할 근거가 생깁니다.
