# TerraDelta preparation plan

Scope: code and non-training tests only. No optimizer steps, GPU instance operations, large imagery downloads, paid resources, or competition submissions.

- [ ] Retrieve official baseline unchanged; record hashes and exact notebook behavior.
- [ ] Implement compatible model/checkpoint, dataset, transforms, postprocessing, and evaluator.
- [ ] Implement external discovery/dry runs, tiling, weak labels, and synthetic/negative generators.
- [ ] Implement step-based training, validation, probability/TTA/alignment inference and search.
- [ ] Build offline submission notebook/exporter, AWS preparation, licenses, and documentation.
- [ ] Run CPU tests including official checkpoint forward and extracted submission execution.
- [ ] Review, commit meaningful stages, push verified code, and report limits and next commands.

## Invariants

Paired geometry is identical. Classes are background/new_building/tree_removal. Geographic split groups never overlap. Official weights never trigger network downloads. Unknown licenses cannot enter default training. Polygon evaluation uses exported exteriors. ZIP entry point is at root. Original baseline is immutable and excluded from Git where redistribution is unconfirmed.
