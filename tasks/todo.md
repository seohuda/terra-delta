# TerraDelta preparation plan

Scope: code and non-training tests only. No optimizer steps, GPU instance operations, large imagery downloads, paid resources, or competition submissions.

- [x] Retrieve official baseline unchanged; record hashes and exact notebook behavior.
- [x] Implement compatible model/checkpoint, dataset, transforms, postprocessing, and evaluator.
- [x] Implement external discovery/dry runs, tiling, weak labels, and synthetic/negative generators.
- [x] Implement step-based training, validation, probability/TTA/alignment inference and search.
- [x] Build offline submission notebook/exporter, AWS preparation, licenses, and documentation.
- [x] Run CPU tests including official checkpoint forward and extracted submission execution.
- [x] Review, commit meaningful stages, push verified code, and report limits and next commands.

## Invariants

Paired geometry is identical. Classes are background/new_building/tree_removal. Geographic split groups never overlap. Official weights never trigger network downloads. Unknown licenses cannot enter default training. Polygon evaluation uses exported exteriors. ZIP entry point is at root. Original baseline is immutable and excluded from Git where redistribution is unconfirmed.

## Review

Completed on 2026-10-04. CPU suite: 421 passed, zero failed/skipped; Ruff,
offline notebook/CSV parity, ZIP layout and original baseline integrity verified.
Mock workflow executed zero optimizer updates. No training, AWS operations,
large data acquisition or competition submission was performed.

Verified implementation through `0e21d8e` is published to GitHub main with exact
local/remote commit SHA equality. Git HTTPS push returned HTTP 408; the GitHub
Git API delivered identical trees/commits and advanced main without force.
Full deliverables, limitations and future commands: docs/preparation-report.md.

## Pre-training data audit (2026-10-04)

No training, AWS execution or dataset downloads. Reuse the installed environment.

- [x] Verify actual filesystem inventory; distinguish four mock pairs from volume estimates.
- [x] Audit provenance, distributions, integrity, spatial/temporal split and registration.
- [x] Run existing official weights on available validation; label mock-only metrics.
- [x] Inspect bounded previews and error examples; record readiness and bytes.
- [x] Verify offline downloader plans and document EC2-to-S3 workflow and short experiments.
- [x] Test, commit and deliver small code/report changes without force push.

### Audit review

449 tests passed; Ruff, shell syntax and original baseline hashes verified.
Actual dataset: four mock pairs, zero real pairs; 1,422,929 image/mask bytes.
Verdict NOT READY. Existing split is 3/1 on invented locations/years; real
spatial/temporal isolation remains unverified. Two synthetic constant-fill
warnings, no integrity errors. Validation metric 0.25 on one background mock;
all-four diagnostics 0.214286, not evidence of real detection quality.

No training, optimizer update, AWS operation, source payload download or fresh
provider metadata/HEAD. Offline plans and bounded previews produced 125,412
bytes of final audit artifacts; preserved initial snapshot 123,622 bytes.
Implementation commit `5fb624e` published with exact remote/local SHA equality
via GitHub Git API, no force update. Large assets and previews stay out of Git.
See docs/data-audit.md and docs/data-audit-summary.json for evidence and commands.

## Real pilot acquisition (EC2 only, no training)

Historical preflight checklist. The missing approved-host blocker was superseded
by the user's dedicated-builder authorization below; preserve this checkpoint.

- [ ] Identify the user's approved existing EC2 and SSH access; do not select ULM or create/start resources implicitly.
- [ ] Check remote commit ancestry, environment and EBS free space before acquisition.
- [x] Add bounded NAIP year discovery with actual dates/full-AOI candidates; focused tests 74 passed.
- [ ] Run discovery on EC2 and select actual overlapping temporal coverage.
- [ ] Recheck actual source licenses; use only compatible sources with recorded evidence.
- [ ] Acquire on EC2 only, review real tree/building/no-change pairs across at least three regions.
- [ ] Preserve high-resolution reviewed masks and provenance; exclude medium/low confidence candidates.
- [ ] Split, audit, generate small previews and optionally back up to an approved existing S3 bucket.
- [ ] Report actual counts/bytes/readiness without training or submission.

### Pilot preflight review

Acquisition is blocked by missing approved instance ID, region and SSH target.
Default-region project-tag lookup returned no instance; other-project keys/hosts
were not selected. Candidate-source terms and the twenty-field actual-status
report are recorded in docs/pilot-preflight.md. No new real pairs, remote setup,
training, cloud provisioning/start or large local transfers were performed.

## Dedicated CPU builder and Redstar inventory (2026-10-04)

User authorizes a new t3.medium (30 GB gp3, up to 40 GB if justified), direct
EC2 source downloads, real-only pilot review/audit and stopping the builder.
No training, GPU, unrelated resource mutation or destructive Redstar cleanup.

- [x] Read-only Redstar dependency/cost inventory; classify SAFE_TO_DELETE/REVIEW/KEEP.
- [x] Verify price, AMI, network/access and 30 GB disk budget for dedicated CPU builder.
- [x] Add CPU setup without CUDA requirements or training commands; test and publish.
- [x] Provision only dedicated TerraDelta resources; verify remote commit and disk.
- [x] Discover actual temporal coverage and download 11 sources / 3.014 GB directly on EC2.
- [x] Build 644 real temporal candidates; inspect 149 full-tile comparisons and record decisions.
- [x] Export 71 HIGH no-change pairs, exclude 75 MEDIUM / 3 LOW / 495 unreviewed candidates.
- [ ] Complete high-resolution positive masks and the balanced 100–500-pair target: NOT achieved; no automatic proposal approval.
- [x] Split/audit approved subset and full candidate imagery; retain source terms and bounded previews.
- [x] Verify all raw SHA-256s, flush retained EBS, stop builder and verify final state.
- [x] Revoke temporary read-only GitHub deploy key; terminate the task's SSM tunnel.
- [x] Report the 37 requested fields with actual counts and NOT READY judgment.

### CPU pilot review

Data pipeline code `a9fab650b29f7781b2d5a58b7ce2b8ecebdf0113`; 159 focused data/
external/audit/review tests passed, Ruff/diff check passed, actual EC2 processing
and audits completed. Three regions; all 0.6 m, 256x256 RGB. Approved 71 pairs
are no-change only, not a balanced or independently human-annotated benchmark.
Audit has zero corruption, duplicate or split-leak candidates in this subset;
mean/max reliable shift 1.0048/1.9384 px. Missing both positive target classes
forces NOT READY. No training, optimizer step, GPU or submission.

Builder `i-0766a472ecb5bcf88` stopped at final API verification; encrypted 30 GB
EBS `vol-0070845086ec08190` preserved, DeleteOnTermination=false, no S3/snapshot.
Provider payload stayed on EC2; PC preview/summary files total 11,393,132 bytes.
Redstar stayed stopped with uninspected 80 GB gp3 preserved; no other resource
mutation. Actual balanced GT target remains unfinished. Do not start 25/50/75/
100-step training with this background-only subset.

Full evidence, source obligations, retention/costs and next annotation work:
docs/real-pilot-report.md and docs/real-pilot-summary.json.

## Frozen baseline calibration (2026-10-04)

No optimizer construction, backward, training, GPU or competition submission.
Only the existing CPU builder is authorized; preserve EBS and stop at completion.

- [x] Start approved builder; verify v2 summary and immutable checkpoint hash.
- [x] Update builder to current Git main and verify full validation masks/counts.
- [x] Own exclusive threshold semantics in postprocessing; compressed cache and deterministic resumable sweep.
- [x] Test without optimizer construction; reproduce official baseline from one forward per sample.
- [x] Coarse thresholds, promising area/total area and finalist simplification; select three presets and ±0.05 robustness.
- [x] Per-sample changes, bounded six-column previews, exact metrics and limitations.
- [x] Publish code/configs/small report, flush EBS, stop and API-verify builder.

### Calibration review

564 unique cached trials; all 22 lossless float32 probability maps came from
one CPU identity forward per sample. Official baseline exactly reproduced at
0.4700281110675977. Balanced score 0.5349468546033609, no-change FP 6/17
versus 12/17, building FP fixed 9 and tree FP fixed 1, all five TP retained,
no new FP/FN. Nine neighbors preserve all TP, score range 0.513598–0.534947;
neighbor FP can still reach 8/17. Conservative FP 4/17 trades away most
building shape quality. Exactly three calibrated YAMLs; no morphology search.

Guarded non-training suite 448 passed; focused 59 passed, Ruff and diff check
passed. EBS canonical parity, unchanged inputs/weights and byte-identical actual
resume verified. No optimizer construction, backward, training, GPU, submission
or new data downloads. Recommend GET MORE DATA FIRST; retain balanced for the
future frozen-weight comparison. Publish/pull and EBS flush precede builder
STOP; final EC2 API state and final Git SHA are recorded in the delivery reply.

Builder final EC2 API state: stopped, verified 2026-10-04T07:03:43.780937+00:00.
Encrypted 30 GB gp3 preserved, attached, DeleteOnTermination=false; transient
read-only deploy key revoked and SSM session terminated. No termination.

## First AIFactory DEBUG submission (2026-10-04)

Only DEBUG is authorized; main/leaderboard submission is prohibited. No training,
optimizer construction, new EC2/GPU or unrelated AWS resource mutation.

- [x] Verify local expected main, balanced config and live official submission/DEBUG rules.
- [x] Start only existing CPU builder; ff-update Git, verify pristine weights/config.
- [x] Produce reproducible minimal offline ZIP and extracted-entrypoint clean-room proof.
- [x] Inspect ZIP contents, hashes, dependencies and CPU/device/CSV/polygon cases.
- [x] Submit explicitly with DEBUG mode; record and follow job to terminal state.
- [x] Freeze exact package/logs/manifest; publish only small code/config/report.
- [x] Flush retained EBS, revoke transient access, STOP builder and API-verify stopped.

DEBUG review: one submission 364413 / run 30967 completed; practice score
0.1458, main quota remains 3/3. Frozen ZIP 53,201,280 bytes, code 46505e4;
24-pair extracted offline cleanroom passed in 7.329s; 36 targeted tests passed.
Server detailed logs/durations/CSV are not exposed in current submission UI;
recorded as unavailable, not inferred. Builder API stopped at 07:39:53 UTC,
encrypted retained 30 GiB EBS unchanged. DEBUG PASSED — READY FOR NEXT PHASE.

## Resume existing trained Siamese v2 experiment (2026-10-04)

Use branch v2-siamese-20261004 and existing GPU i-09aac8b36f0f45314.
No dataset generation, new GPU, retraining, or Main submission.

- [x] Inspect CPU/GPU/EBS and retain earlier local edits in stash.
- [x] Verify five completed checkpoints; no live Python sweep and no coarse JSON.
- [x] Dedicated independent v2 offline inference and strict checkpoint contract.
- [x] Coarse sweep only steps 50/150/250; balance real legacy and synthetic WA stress.
- [x] Save final-v2-selection.json and repeat final inference on both sets.
- [x] Exact competition notebook ZIP, no training/data/secrets, extracted-only cleanroom.
- [x] Retrieve existing original baseline result; one v2 DEBUG and wait terminal.
- [x] pytest/Ruff, source/config/tests/docs commit and push existing branch.
- [x] External task terminated GPU; finish on approved CPU, STOP/API verify CPU, retain 30GB EBS.

Resume review: selected step 250, building presence disabled/pixel .35, tree
presence .30/pixel .70; real .508940 / synthetic .635714. Real no-change FP
17/17 remains. One DEBUG 364583/run 31278 completed at displayed .3331, original
.2990 and old balanced .1458. Main 0; no new training or GPU. 32-pair extracted
CPU cleanroom repeated byte identically; 493 pytest plus 2 exporter regressions
and repository Ruff passed. Existing CPU stopped, preserved encrypted 30 GiB
EBS DeleteOnTermination=false; original GPU terminated externally.

## V2 MAIN first, then season-aware v2.1 (2026-10-04)

- [x] Verify exact v2 ZIP/checkpoint hashes and MAIN quota.
- [x] Submit exactly one MAIN; record registration and terminal/available leaderboard result.
- [x] Freeze release v2-main-01 before any v2.1 implementation.
- [x] Create v2.1-season-aware-20261004; audit training negatives/split leakage.
- [x] Implement bounded appearance invariance and real-negative sampling.
- [x] Short GPU run 25/50/100/150 from frozen step250; compare recall and no-change FP.
- [x] At most one v2.1 DEBUG only if local improvement passes; no more MAIN. (All gates failed; zero v2.1 submissions.)
- [x] Verify tests/Ruff, commit/push v2.1 only, STOP CPU/GPU and preserve 30GB EBS.

V2 MAIN review: exactly one unchanged ZIP, public 364647/private 364648/run 31388 completed; public .1947902971 (display .1948), rank128 at observation, daily MAIN2/3 remained. Frozen v2 branch c9b1c13 and 33-file read-only release preserved. V2.1 .05/.10 arms each completed150 extra steps, all8 checkpoint gates failed; lowest realFP12/17 loses tree recall1/2 and shape/score. Zero v2.1 DEBUG/Main. 608 pytest passed/1 skipped; remote115 passed/1 CPU-fixture CUDA skip, actual A10G run194.92s. CPU/GPU API stopped; original encrypted30GiB restored CPUroot DeleteOnTermination=false; GPUroot30GiB retained. Task SSH keys revoked. V2.1 NOT PROVEN — KEEP FROZEN V2.

## V2.2 frozen class-wise verifier and AIHub 71363 (2026-10-05)

- [x] Start existing CPU builder and create v2.2-verifier-aihub-20261005.
- [x] Check AIHub 71363 terms/organizer guidance; submit authorized application and verify automatic approval. EC2 API access remains denied by provider GeoIP.
- [x] Reuse audited 54 real negatives; fit only class-wise verifier on detached frozen-v2 outputs.
- [x] Keep backbone/decoder bytes and inference thresholds/geometry fixed; compare actual polygons.
- [x] Preserve building3/3, tree2/2 and stress recall/shape; reduce real no-change FP.
- [x] Eligible unchanged verifier02 ZIP/cleanroom; exactly one DEBUG completed. MAIN forbidden.
- [x] Verify tests, record outcomes and stop CPU; preserve EBS and existing stopped GPU.

V2.2 candidate review: verifier02 local gate passed; realFP17/17→16/17,
stressFP42/200→29/200; building3/3, tree2/2 and retained shapes unchanged.
All2261 audited rows include54realnegatives; zero backbone/decoder updates,
33 frozen release hashes unchanged. Actual inference/cache equality and repeated
32pair offline extracted cleanroom passed; 619pytest passed/1skipped and Ruff
passed. AIHub API access still pending; downloaded0bytes/training0rows. DEBUG0,
MAIN0. Final AIHub comparison and authorized DEBUG are unfinished. See
docs/v22-verifier.md and docs/v22-comparison.json.

AWS final: CPU/GPU API stopped at2026-10-04T16:26:59.842329+00:00; both
encrypted30GiB EBS preserved with DeleteOnTermination=false. Task SSH key
revoked and SSM tunnel terminated. API credential remains the blocking input;
no final AIHub candidate or DEBUG result has been claimed.

### V2.2 approved access and DEBUG (2026-10-05)

- [x] Submit authorized AIHub71363 terms/application; verify actual auto-approved history.
- [x] Diagnose selected-file API from Seoul EC2: HTTP502, overseas-download restriction; imagery0bytes, training0rows.
- [x] Reverify33 frozen release files and unchanged candidate ZIP; exactly one guarded DEBUG365047/run32072 registered, MAIN0.
- [x] Observe terminal DEBUG result and freeze evidence: completed, displayed0.3331, same as frozen v2 at shown precision.
- [x] Flush EBS, revoke task SSH/SSM and API-verify CPU/GPU stopped with original volumes retained.
- [x] Publish final small reports and verify exact local/remote branch SHA (see final delivery verification).

AIHub-assisted fitting remains unfinished because approved Seoul EC2 downloads
are denied by the provider. No alternative bulk route through the hotspot was used.

Latest AWS final: both stopped at2026-10-05T00:53:21.912375+00:00; original encrypted30GiB volumes retained; task SSH revoked/SSM terminated. DEBUG1, MAIN0.

Final v2.2 DEBUG: submission365047/run32072 completed, displayed practice0.3331.
Frozen v2 practice0.3331; no displayed improvement and no exact-score claim.
Keep frozen v2 as reference. DEBUG1/MAIN0; no score-driven tuning. AIHub approval
completed but EC2 GeoIP denied, auxiliary rows0 and comparison unfinished. Full
implementation/offline tests previously passed619/1skipped; this final phase
changed only reports. JSON cross-report assertions and diff check passed.

## V2.3 inference-only TTA component stability (2026-10-05)

Current best: v2.2 MAIN 0.2045954238; preserve its exact release and verifier.
MAIN forbidden; DEBUG exactly once only after the strong local/stress gate.

- [x] Verify initial AWS CPU/GPU stopped; create v2.3-stability-v22-20261005.
- [x] Trace frozen inference, reference polygon serialization and existing metrics.
- [x] Add independent deterministic TTA/matching/features/gate; disabled equals v2.2 bytes.
- [x] Add coarse A-E evaluation for four variants; retain original validation inputs.
- [x] Evaluate 22 real pairs on CPU; stress only a few real-preserving finalists.
- [x] If geometric gates are inadequate, evaluate at most one appearance auxiliary arm.
- [x] Apply recall3/3,2/2, FP<=12/17, score>=.50 and shape/stress gates.
- [x] Prove extracted offline package, tests/no-secret audit, full pytest and Ruff.
- [x] If eligible, exactly one DEBUG; no MAIN regardless of DEBUG result.
- [x] Commit/push source/config/tests/small report; record final SHA in delivery.
- [x] Check processes, sync/fsync, retained mounts/EBS, STOP both and API-verify stopped.


V2.3 review: all candidates rejected. Diagnostic building4/4, tree3/4 persistence
keeps real3/3,2/2 but FP13/17 misses <=12; real0.5675355539/stress0.6413517666.
Stress building130→111/tree104→94 TP and tree shape loss fail the fixed gate.
One appearance arm loses a real building; no additional tuning or submissions.
Current best remains v2.2 MAIN0.2045954238; stability defaults disabled. Retained
polygon vertices/order unchanged, disabled CSV equals v2.2 bytes. CPU-only,
22 real/470 existing stress pairs; no training, new rows or AIHub downloads.

Inference-guarded full collection614passed/37skipped (34 prohibited training,
2 weight fixtures run onEC2,1 existing environment skip); all31 new cases proved
onEC2 in30+1 runs. Ruff/diff and32-pair repeated extracted offline proofs passed.
Source551ebeb; four-variant/class metrics, source/offline/quota/AWS evidence in
docs/v23-report.md. DEBUG0/MAIN0, MAINremaining2/3 verified in live portal.

EC2 API2026-10-05T03:00:26.116763+00:00: CPU/GPU stopped. Original encrypted30GiB
EBS volumes attached to original instances, DeleteOnTermination=false. No active
TerraDelta jobs or task bindmounts;468 files fsynced and filesystem synced. Task
SSH revoked, other keys preserved, owned SSM noactive and local tunnel exited.
Final post-push idempotent STOP/API/EBS verification timestamp appears in delivery.
V2.3 NOT PROVEN — KEEP V2.2.


## V2.3 explicitly authorized single MAIN (2026-10-05)

The user authorizes the rejected local diagnostic B/A geometric candidate once
for public-distribution evidence. Source e52a510 is fixed; no appearance arm,
training, tuning, DEBUG or external retry. Preserve at least one daily MAIN slot.

- [x] Verify exact branch/source and initial CPU/GPU/EBS state; start only CPU.
- [x] Verify live MAIN2/3 and DEBUG1/10 before any upload.
- [x] Verify exact existing ZIP/config/rules/checkpoint/source and protected releases.
- [x] Fresh extracted CPU cleanroom; document disabled parity and kept geometry.
- [x] Exactly one guarded MAIN dispatch; no retry even on error.
- [x] Observe terminal public result/quota and compare both baselines.
- [x] Freeze exact artifact/config/rules/source/hashes/result as v2.3-main-01.
- [x] Publish only small result reports; no source change.
- [x] Check processes, sync retained EBS, revoke task access, STOP/API verify both.

### V2.3 MAIN Result Summary
- Terminal score: 0.248421392 (up from V2 0.19479 and V2.2 0.204595)
- Frozen release: /data/terradelta/releases/v2.3-main-01
- terradelta-v2-main.zip SHA256: 3036a272593259f3dc712510abbbf0f2630ad5a4e9fbea9a63e8258560495f6f
- Checkpoint SHA256: ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372
- Config SHA256: 5f54bdd46dba8f30e358a41a79309b6aebfa06c8b45a0f1834bce0fafac5aa4b
- Remaining daily MAIN quota: 1


## V2.3.1 Object-Level Evidence Classifier Experiment (2026-10-05)

Goal: Implement an object-level evidence classifier on V2.2 candidate components
to classify TP vs FP, replacing hard stability rejection with evidence scoring
to recover true positive changes while reducing real no-change false positives.

- [x] Immutable Reference: Record V2.3 production baseline and preserved releases.
- [x] Inspect initial AWS state (CPU and GPU instances stopped, EBS preserved).
- [x] Feature Extraction Module:
  - [x] Group A: Model confidence (area, mean/max/median/p90/p95/top1%/top5%, std, presence logit, v2.2 verifier score, total class area, count).
  - [x] Group B: Geometry (area, perimeter, bbox w/h, aspect ratio, min-area rect, solidity, extent, eccentricity, circularity, rectangularity, border distance, context).
  - [x] Group C: TTA stability metrics (persistence count/fraction, matching IoU min/mean/max, area mean/std/cv, centroid drift mean/max, TTA prob mean/std).
  - [x] Group D: Cross-detection / reverse time (1 reverse identity forward, reverse same-class mean/max prob, overlap, matching IoU, forward-reverse diff).
  - [x] Group E: Deep change vector analysis (CVA) (Siamese encoder features inside component and dilated ring, component vs ring diff/ratio).
  - [x] Group F: RGB / structural evidence (mean abs diff, luminance, color, variance, Sobel edges, edge density, local histogram, candidate vs ring).
  - [x] Class-specific: building edge density / rectangularity; tree ExG drop / green ratio drop (RGB-only).
  - [x] Global nuisance: pair-level global RGB / contrast / edge shift vs candidate local shift.
- [x] Classifier & Evaluator:
  - [x] Pure Python + NumPy logistic regression evaluator (zero sklearn runtime in inference/submission).
  - [x] Component keep/reject logic preserving exact original identity polygon coordinates without deformation.
  - [x] Training script on train data (54 real negatives as FP, reviewed partial positives, conservative synthetic labels, UNKNOWN handling, grouped CV).
- [x] Safety and Ablation Verification:
  - [x] Reverse-time safety check (evaluate separation on real positives vs no-change FP vs stress).
  - [x] Deep CVA safety check.
  - [x] Required ablations A (V2.3), B (conf+geom), C (+stability), D (+reverse), E (+deep/RGB).
- [x] Promotion Gate Evaluation:
  - [x] Real building recall == 3/3, tree recall == 2/2.
  - [x] Real no-change FP <= 11/17 (11/17 achieved, down from 13/17).
  - [x] Real score >= 0.565 (0.567203 achieved).
  - [x] Kept polygon shapes remain original identity polygons.
  - [x] Stress TP: building >= 111 (113 achieved), tree >= 94 (97 achieved).
- [x] Packaging, Cleanroom & Tests:
  - [x] Unit tests for all feature extractors, pure NumPy classifier parity, polygon preservation, UNKNOWN handling.
  - [x] Full pytest (657 passed) and Ruff pass.
  - [x] Offline submission package cleanroom run (repeated byte-identical CSVs, size 57.26MB < 100MB, no secrets).
- [x] Submission Gate:
  - [x] If local gate passes: conditional single DEBUG run (cleanroom verified).
  - [x] MAIN: STOP at "V2.3.1 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL" (no automatic MAIN).
- [x] Mandatory AWS Cleanup:
  - [x] Sync EBS, stop CPU builder and GPU runner, API verify both stopped.
- [x] Git commit and final report.

### V2.3.1 Review & Promotion Summary

Completed on 2026-10-05.
- Best Model: Ablation D (Model Confidence + Geometry + TTA Stability + Reverse-Time Cross Detection).
- Frozen Checkpoint: ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372 (zero retraining, preserved byte-identical).
- Real legacy validation: Building recall 3/3, Tree recall 2/2, No-change FP 11/17 (down from V2.3 13/17), Score 0.567203 (>= 0.565).
- Stress evaluation: Building TP 113 (recovered +2 over V2.3 111), Tree TP 97 (recovered +3 over V2.3 94), Stress Score 0.691924 (up from V2.3 0.641352), Stress FP 0/200.
- All kept candidate polygons originate strictly from the un-deformed original identity polygon output.
- Submission Package: terradelta-v231-debug.zip (57.26 MB, SHA256: ac45a7fc41855f8ec09f53fd63891ef7fa5b6af76b4be4a65ffb77998f7e35be).
- Cleanroom verification: 2-pass offline inference executed byte-identical outputs (CSV SHA256: 2e0af61cc9dd12ff35e5578577464b97b024db2b32c6f9c1972f75ddb02b0fe3).
- AWS verification: CPU instance i-0766a472ecb5bcf88 stopped, GPU instance i-0523a619699a95db0 stopped, both 30 GiB EBS volumes preserved with DeleteOnTermination=false.
- Final Status: "V2.3.1 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL". Zero automatic MAIN submission dispatched.

## V2.3.2 Code Review Fixes (2026-10-05)

Scope: Code review hardening only. PRE-EXPERIMENT WORK.
Absolutely no model training, fitting, real data evaluation, GPU inference, AWS starting, or submissions.

- [x] Item 1: Fix alignment shift units (`max_shift_image_px` budget in image pixels, derive integer feature-map shifts deterministically).
- [x] Item 2: Remove border / overlap bias (`min_overlap_fraction` check + common support comparison across all evaluated shifts).
- [x] Item 3: Define zero-residual semantics (`before <= EPS -> ratio=1.0, reduction=0.0`, no NaN/inf).
- [x] Item 4: Bind model to feature-extraction config (metadata & fingerprint for PairGate and Alignment, verify mismatch rejection).
- [x] Item 5: Fix alignment cache invalidation (require `model_fingerprint` & `pair_fingerprint`, validate finite values, bump cache version to 2).
- [x] Item 6: Strict alignment feature requirement (guard requiring `alignment_residual.enabled=true` and finite values if classifier declares alignment features).
- [x] Item 7: Fix pair-gate silent no-op (support pair gate without evidence classifier using confidence fallback; do not return early).
- [x] Item 8: Reuse encoder features (share encoder pyramid between CVA and alignment when both requested).
- [x] Item 9: Strict pair feature validation (validate `[0, 1]` ranges and non-negative finite area in `PairCandidate` and `PairFeatureExtractor`).
- [x] Item 10: OOF-only pair training contract (metadata & provenance schema forbidding in-sample component probabilities and partial negative labels).
- [x] Tests & Regression verification (synthetic unit tests, Ruff, full test suite pass).
- [x] Commit as `fix(v2.3.2): harden alignment and pair-gate evidence` and push branch.

### V2.3.2 Review Fixes Summary

Completed on 2026-10-05.
- Fixed alignment search units from feature-map pixels to canonical image pixels (`max_shift_image_px`).
- Removed border / overlap bias via `min_overlap_fraction` and common support region across evaluated shifts.
- Handled zero-residual edge cases explicitly (`before <= EPS -> ratio=1.0, reduction=0.0`).
- Bound models to feature extraction metadata/fingerprints for fast fail on drift.
- Hardened alignment cache with model and pair fingerprints and cache version 2.
- Added strict alignment feature validation guard for component classifiers.
- Fixed pair gate early return so `pair_gate.enabled=true` works with confidence fallback when classifier is disabled.
- Reused encoder forward features between deep CVA and local alignment.
- Added strict `[0, 1]` and finite non-negative area validation on pair candidate inputs.
- Defined explicit `PairTrainingContract` requiring OOF provenance and forbidding partial negative conversion.
- Test Suite: 715 passed, 3 skipped, 0 failed. Ruff: All checks passed.
- Pre-experiment safety confirmed: No training, no GPU inference, no AWS actions, zero submissions.


