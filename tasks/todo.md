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
