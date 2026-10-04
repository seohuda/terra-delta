# First AIFactory DEBUG submission

**DEBUG PASSED — READY FOR NEXT PHASE**

대회 9306에 최종 ZIP 바이트를 EC2에서 직접 한 번 DEBUG 제출했고, 웹 제출 이력의 `완료`와 연습용 점수 `0.1458`을 확인했다. 정규 제출은 없으며 일일 잔여 3/3이 유지된다. 이는 실행·패키징 검증이며 경쟁 성능 평가가 아니다.

| 항목 | 결과 |
|---|---|
| 패키지 Git SHA | 46505e43ca00bbea7d43ec4497f11d568f4b5a94 |
| ZIP 파일명 | terradelta-balanced-debug-v1.zip |
| ZIP bytes | 53201280 |
| ZIP SHA-256 | a681e04d7dd773398048c4eabd1344a06cd0fe5e89c0426960f8824c6d0981fd |
| Checkpoint SHA-256 | 1be127bd2f70c8e3b0efdc48c6d3effc29f30f090746a91c87e6c23e9b3e878b |
| Preset | configs/inference_sweep_balanced.yaml |
| Threshold | building 0.55 / tree 0.30 |
| min_area | building/tree 100 |
| min_pos_area | building/tree 20 |
| simplify_px | building/tree 0.25 |
| Backend / ndigits | reference / 2 |
| Semantic | 임계값을 통과한 양성과 배경의 최대 확률. 배경 동률 우선; 양성끼리 동률이면 building. 클래스 겹침 없음. |
| Morphology | opening/closing/dilation/erosion 0, fill_holes false |
| Inference | identity TTA, alignment none, batch_size 1, encoder_weights null; UNet/ResNet18 6-channel RGB/ImageNet |
| Clean-room | PASS: 추출 ZIP 코드만 -I로 실행, CPU 24쌍(실제 22+합성 2), 네트워크/optimizer/backward 차단, 7.329초; 기존 CPU 의존성 환경 재사용 |
| DEBUG 제출 횟수 | 1회; 재제출 0회; DEBUG 1/10 |
| DEBUG job | debugSubmissionId 364413 / codeRunId 30967 |
| DEBUG 상태 / 시간 | 완료; 2026-10-04 16:33:05 KST 접수, 16:41:17 KST 완료 확인. 서버 설치/추론/대기 시간은 UI 미제공. |
| 로그 / 오류 | 로컬 stderr 비어 있음. 서버 stdout/stderr/설치/추론 로그와 경고·OOM·timeout 상세는 현재 UI 미제공; 독립 확인 불가. |
| Output validation | 로컬 ID/행수/컬럼/JSON/유한 좌표/0–256 범위/유효 polygon/빈 결과/두 클래스 PASS. 서버는 채점 완료·점수 생성; 원본 CSV 미제공. |
| DEBUG 점수 | 0.1458 (화면 4자리 표시, 연습용; 리더보드 반영 없음) |
| Main 제출 / quota | 0회, 일일 정규 횟수 소비 0, 잔여 3/3 유지 |
| Optimizer / backward / GPU training | 모두 0회. 새로운 EC2/GPU 생성·시작 없음; 기존 CPU 빌더만 사용. |
| EC2 / EBS | i-0766a472ecb5bcf88 stopped(API 확인). 암호화 30 GiB EBS 유지, DeleteOnTermination=false; 임시 Git 키 폐기, SSM 종료. |
| 다음 단계 | DEBUG PASSED — READY FOR NEXT PHASE. 패키지 고정; 별도 승인된 다음 단계로 이동. 점수 개선 판단은 기존 GET MORE DATA FIRST 유지. |

최종 ZIP은 EBS의 `/data/terradelta/submissions/balanced-debug-v1/terradelta-balanced-debug-v1.zip`에 mode 0444로 보존된다. PC/Git로 가중치·데이터·ZIP을 중계하지 않았다. 동일 패키지 ZIP을 두 번 만들었고 SHA-256이 같았다. 22개 엔트리의 경로/크기/개별 SHA는 [manifest](debug-submission-manifest.json)에 있다. 원본 체크포인트와 실제 입력 44개 파일의 해시도 변하지 않았다. 패키지 코드 SHA와 이후 보고서 커밋은 구분한다.

패키징 변경은 배포에 불필요한 calibration 모듈 제외와 고정 ZIP 메타데이터다. 압축에는 파일 스트림을 사용한다. `tests/test_submission.py tests/test_postprocess.py` 36 passed, Ruff와 diff check passed. 이 변경 이후 최종 ZIP으로 클린룸 검증을 수행했다. 추론 코드/가중치는 제출 뒤 수정하지 않았다.

현재 [제출 화면](https://aifactory.space/ko/competitions/9306/submission)은 DEBUG 완료와 점수만 보여 주며 상세 로그·실행 시간·서버 CSV를 노출하지 않는다. 따라서 이를 추정하거나 무경고를 주장하지 않는다. 로컬 실행 시간 7.329초와 접수부터 완료 확인까지의 약 8분 12초는 서로 다르며, 후자는 큐와 설치를 포함한 관측 상한이다. EBS manifest에는 STOP 전 저장된 패키지/클린룸/접수 증거가 있고, 종료 확인·shutdown 정보는 이 저장소의 최종 manifest에 보존한다. 서버 로그 수집만을 위해 빌더를 다시 시작하지 않았다.

참여 키는 stdin을 통해 EC2 프로세스에 일시 전달했다. 파일·환경 파일·Git·ZIP·로그에 저장하지 않았다. 공식 aifactory 3.2.2 SDK의 기존 ZIP 전송 함수를 사용해 CLI의 임시 재패키징을 피했다. 업로드 URL 발급과 등록 요청 모두 `debug: true`를 검사했다. SDK는 추론 requirements에 포함되지 않는다. [공식 SDK](https://pypi.org/project/aifactory/)와 [대회 규정](https://aifactory.space/ko/competitions/9306)을 현재 확인했다.
