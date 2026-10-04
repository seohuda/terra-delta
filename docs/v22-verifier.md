# Frozen v2.2 verifier candidate

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
Selected verifier thresholds: building0.015, tree0.05. This fixes only one real
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
file inventory and the required label/split treatment. The user logged in; the
separate AIHub API key for direct provider-to-EC2 acquisition remains pending.
AIHub imagery downloaded: 0 bytes. AIHub training rows: 0. No static segmentation
label has been relabeled as temporal change. The final auxiliary-data comparison
is therefore unfinished.

DEBUG submissions: 0. MAIN submissions: 0. The authorized single DEBUG is reserved
until AIHub access and sample suitability are resolved. No server score is claimed
for v2.2. Frozen v2 remains the reference (DEBUG0.3331; MAIN0.1947902971).

## AWS state

The CPU and GPU were both API-verified stopped at
2026-10-04T16:26:59.842329+00:00 (October5 KST). Both encrypted30GiB root
volumes remain attached with DeleteOnTermination=false. The task SSH key was
removed and the task SSM forwarding session terminated. Source, feature caches,
original data, frozen weights and the candidate ZIP are preserved. See
[v22-aws-final.json](v22-aws-final.json).

## Resume

Read small reports/config/source already committed on
`v2.2-verifier-aihub-20261005`. Start only the existing CPU builder
`i-0766a472ecb5bcf88` when the pending AIHub credential is available. Download
selected files directly on Seoul EC2 with explicit byte/disk limits, inspect
actual imagery/labels/provenance, and fit only the verifier on compatible audited
auxiliary examples. Keep candidate02 as a fallback if auxiliary data fails the
local gate. Finalize once, execute exactly one guarded DEBUG, inspect its terminal
result, and stop CPU while retaining EBS. MAIN remains prohibited.
