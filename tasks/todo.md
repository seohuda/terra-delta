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
