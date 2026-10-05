# Frozen v2.2 verifier result

The class-wise verifier preserves the exact frozen v2 backbone, decoder, pixel
thresholds and polygon geometry. It independently suppresses a class candidate
when a regularized linear model rejects it. A feature outside the observed
training range causes abstention: retain the original frozen prediction.

## Training and selection

The verifier fits detached, fixed image/probability statistics with NumPy IRLS.
It uses all 2,261 existing audited training rows, including all 54 real no-change
pairs. Partial-label absence is excluded. Real and synthetic strata are weighted
within each class; no torch optimizer or backward is executed. The 33-file frozen
release and checkpoint hashes remain unchanged.

All five real validation positives fall outside at least one training feature
range. The initial linear verifier rejected positives and failed the gate.
Training-support abstention retains those frozen predictions. The same 22 real
WA and 470 synthetic WA pairs guide design/threshold selection, so the result
is an adaptive local diagnostic, not independent generalization evidence.

| Population | Frozen score | Verifier score | Frozen no-change FP | Verifier no-change FP |
| --- | ---: | ---: | ---: | ---: |
| Real WA, 22 pairs | 0.5089401837 | 0.5226901837 | 17/17 | 16/17 |
| Synthetic WA, 470 pairs | 0.6357135503 | 0.6439239261 | 42/200 | 29/200 |

Real building recall remains 3/3 and tree recall 2/2. Synthetic building TP stays
130/150 and tree TP 104/150. Shapes of retained predictions remain unchanged.
Selected verifier thresholds: building 0.015, tree 0.05. This fixes only one real
no-change pair; it does not establish competitive generalization.

## Artifact and proof

The candidate is retained on the existing Seoul CPU builder's preserved EBS:
`/data/terradelta/v2.2/debug/verifier-02/terradelta-v2-debug.zip`.
ZIP SHA256: `f4bcaa54157fe077318f9c0bef43d764bab3c14daa86c52a1f566213169d79c2`;
60,019,810 bytes. Frozen checkpoint SHA256:
`ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372`.

Actual inference on both validation populations exactly matches cached candidate
polygon metrics. Two reproducible archives have identical hashes. The extracted
notebook runs twice on 32 pairs with byte-identical CSV, network/optimizer/backward
blocked, CPU fallback and schema/ID/polygon/class-independence checks passed.
Final local tests: 619 passed, 1 skipped; Ruff and whitespace checks passed.
The prepared DEBUG registration guard was tested using a fake client: zero real
API calls, MAIN/wrong endpoints/duplicate registration blocked.

## AIHub and submission status

[AIHub review](v22-aihub-review.md) records organizer guidance, provider terms,
file inventory and label/split requirements. The application is now automatically
approved with the user's explicit terms authorization. Direct selected-file API
requests from Seoul EC2 are nevertheless denied with the provider's overseas
IP restriction message. [Access evidence](v22-aihub-access-final.json) records
that response. Original imagery downloaded: 0 bytes; auxiliary training rows: 0.
Actual band/label/geography suitability and AIHub-assisted fitting could not be
completed. No static segmentation label became temporal change ground truth.

The unchanged verifier02 package was selected because its existing local gate
and offline cleanroom passed. Exactly one DEBUG was registered on
2026-10-05T00:49:26.546621+00:00: submission 365047, run 32072.
Both SDK calls explicitly carried debug=true. MAIN submissions for v2.2: 0.
The portal confirms one daily DEBUG used and all 3/3 daily MAIN slots remaining.
The DEBUG completed with displayed practice score **0.3331**, equal to frozen
v2 at the displayed precision. The unrounded score, detailed server logs,
prediction CSV and execution duration are not exposed in the observed UI.
The local no-change reduction did not produce a displayed DEBUG improvement.
See [v22-debug-result.json](v22-debug-result.json) and the updated
[v22-debug-manifest.json](v22-debug-manifest.json). Frozen v2 remains the
reference (DEBUG 0.3331; MAIN 0.1947902971). The adaptive local improvement alone
does not justify a MAIN submission.

## AWS state

At 2026-10-05T00:53:21.912375+00:00, AWS APIs confirmed the CPU and GPU both stopped.
The task SSH key was revoked and the task SSM forwarding session terminated.
The two original encrypted 30 GiB volumes remain attached and retained with
DeleteOnTermination=false. Before shutdown, all 33 frozen release files and the
registered DEBUG archive hash were verified unchanged; original data, weights,
features and package remain on EBS. See [v22-aws-final.json](v22-aws-final.json).
The initial shutdown and subsequent CPU restart are historical; this is the
latest verified final state. No GPU start, new instance, new volume, resize or
snapshot occurred in v2.2.

## Decision and remaining limitation

Keep frozen v2 as the reference. Verifier02 remains a reproducible local-gate
candidate with a completed single DEBUG and no displayed practice improvement.
No tuning was performed using DEBUG feedback. No MAIN was submitted for v2.2.
All executable work is complete; AIHub-assisted fitting and comparison remain
unfinished because approved Seoul EC2 downloads are denied by the provider.
Resolving provider access is required before an actual auxiliary-data experiment.
The frozen model and the submitted candidate are both retained for later review.
