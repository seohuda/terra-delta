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
