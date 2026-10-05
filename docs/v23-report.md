# V2.3 NOT PROVEN — KEEP V2.2

Current best remains **v2.2 MAIN 0.2045954238**, source `099c1e9`. The inference-only
component gate preserves all five real positives but only reduces real no-change
FP from **16/17 to 13/17**, missing the required <=12/17 promotion. Stress building
TP falls 130→111 and tree TP 104→94; both exceed the predeclared 5% loss bound.
No DEBUG or MAIN was submitted. No thresholds were adjusted after these results.

## Four required variants

| Variant | Building recall | Tree recall | Real no-change FP | Real score | Stress score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen v2 | 3/3 | 2/2 | 17/17 | 0.5089401837 | 0.6357135503 |
| V2.2 verifier | 3/3 | 2/2 | 16/17 | 0.5226901837 | 0.6439239261 |
| V2 stability-only | 3/3 | 2/2 | 13/17 | 0.5675355539 | 0.6407933696 |
| V2.2 + stability | 3/3 | 2/2 | 13/17 | 0.5675355539 | 0.6413517666 |

22 real pairs: 17 no-change, 3 building, 2 tree. Stress: 470 existing synthetic
pairs with 200 no-change and 150 positives for each class. The local selection
reuses these 22 real pairs; it is an adaptive diagnostic, not independent
generalization evidence. All original input files and both baselines were
hash-checked/reproduced unchanged. Exact results: [comparison JSON](v23-comparison.json).

## Real class metrics

Presence F1 below is the published **macro F1** (positive/negative average).

| Variant | Class | TP | FP | FN | TN | Presence macro F1 | Shape score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen v2 | new_building | 3 | 19 | 0 | 0 | 0.1200000000 | 0.8826455217 |
| Frozen v2 | tree_removal | 2 | 10 | 0 | 10 | 0.4761904762 | 0.5569247367 |
| V2.2 verifier | new_building | 3 | 18 | 0 | 1 | 0.1750000000 | 0.8826455217 |
| V2.2 verifier | tree_removal | 2 | 10 | 0 | 10 | 0.4761904762 | 0.5569247367 |
| V2 stability-only | new_building | 3 | 15 | 0 | 4 | 0.3167701863 | 0.8855493140 |
| V2 stability-only | tree_removal | 2 | 9 | 0 | 11 | 0.5086848635 | 0.5591378517 |
| V2.2 + stability | new_building | 3 | 15 | 0 | 4 | 0.3167701863 | 0.8855493140 |
| V2.2 + stability | tree_removal | 2 | 9 | 0 | 11 | 0.5086848635 | 0.5591378517 |

The shapes/vertices of retained polygons are unchanged. Removing extra identity
components slightly raises aggregate real shape scores; it does not reconstruct
or smooth the retained polygons.

## Rules, coarse selection and rejection

Four deterministic paired transforms: **identity, horizontal flip, vertical flip,
180° rotation**. Auxiliary probability maps are inverse-restored to original
coordinates. No mean-probability segmentation mask is created. Inference costs
**4 frozen forwards** per nonempty candidate batch (up to 4× v2.2); batches already
rejected by v2.2 can bypass auxiliary forwards.

Matching is independent per class and one-to-one. Fixed, pre-validation criteria:
IoU >=0.20 OR centroid distance <=4 pixels with area ratio in [1/4,4]. Prefer IoU,
then centroid distance, then reference component indices. The same matching rule
is used for building and tree; it was not tuned on held-out examples.

Persistence includes identity. Mean/minimum IoU use the three auxiliary views
(missing matches contribute zero), avoiding identity’s automatic IoU=1 bias.
Probability moments/consensus pool all four maps over the original component.
Area and centroid moments use matched views. `largest_confidence` means identity
mean probability of the largest eligible same-class component. All requested
features are recorded on EBS; area/drift/probability diagnostics were not turned
into additional fitted thresholds.

Only A–E were evaluated, followed by one class-specific combination. The diagnostic
plateau uses **building: 4/4 persistence (B)**; **tree: >=3/4 persistence (A)**.
Tree presence metrics tie across A–E within the 0.01 score plateau, so its weakest
rule was retained. Building had no neighboring passing plateau beyond B. This
limited evidence does not justify deployment. The checked-in overlay defaults
to **stability disabled**, preserving v2.2 output bytes.

| Coarse rule | Combined real FP | Combined real score | Result |
| --- | ---: | ---: | --- |
| A | 14/17 | 0.5565248255 | Reject |
| B | 13/17 | 0.5680046363 | Reject |
| C | 14/17 | 0.5565248255 | Reject |
| D | 14/17 | 0.5565248255 | Reject |
| E | 14/17 | 0.5569939079 | Reject |
| class_selected | 13/17 | 0.5675355539 | Reject |
| appearance_once | 13/17 | 0.4907433037 | Reject |

A: persistence>=3/4; B: 4/4; C/D/E: >=3/4 plus mean auxiliary IoU >=0.20/0.30/0.40.
No Cartesian sweep or fitted model was added. Neither class-specific nor B/B
cleared FP<=12. Only these two real-preserving finalists received stress evaluation.
B/A stress retains 111/150 building TP and 94/150 tree TP versus v2.2’s 130/150
and 104/150. Tree shape falls 0.5297513736→0.4873021879. Stress no-change FP becomes
0/200, but the positive/shape collapse makes this a failure.

Geometric failure permitted exactly one extra appearance arm: per-channel PRE
mean/std normalization to POST (scale clipped to [0.5,2]), auxiliary inference
only. Original identity components still supply every polygon. This 5-forward
arm loses one real building (2/3), lowers building shape to 0.5595347907 and real
score to 0.4907433037. It was rejected immediately and received no stress run.
No histogram/normalization threshold search or additional appearance arm followed.

## Tests, package and immutable baseline

Full collection under the mandatory inference guard: **614 passed, 37 skipped**.
34 existing tests require optimizer/backward/verifier fitting and were explicitly
skipped; two exact-weight package tests execute on EC2 rather than downloading
weights locally; one existing environment-dependent test remains skipped.
An initial guard run caught two legacy Trainer constructors before optimizer
construction; their explicit exclusions were added without changing legacy tests.
No v2.3 test was excluded for training. EC2 ran 30 v2.3 tests including both
extracted notebook packages, plus the final metrics-adapter test (1 passed).
**Ruff and whitespace checks passed**.

Commands:

```sh
PYTHONPATH=tests:src .venv/bin/pytest -p inference_only -q
.venv/bin/ruff check .
# On CPU EC2, retain weights locally there:
TERRADELTA_FROZEN_V2_CHECKPOINT=/data/terradelta/releases/v2.2-main-01/model.pt \
  PYTHONPATH=tests:src python -m pytest -p inference_only -q tests/test_stability_v23.py
```

The real selected diagnostic ZIP contains 26 entries and runs twice in an
extracted CPU cleanroom on 32 pairs. Network, optimizer and backward are blocked;
imports must resolve to extracted code only. Repeated CSVs are byte-identical,
IDs/schema/bounded valid polygons pass, and every kept polygon’s coordinates
and order are an exact subset of original v2.2 identity output. Disabled package
CSV is byte-identical to v2.2. ZIP reproducibility/no-secret inventory checks pass.
This ZIP is an **offline test of a rejected candidate**, not a submission-ready
release. Weights/data/ZIPs stay on EBS and are excluded from Git.

Frozen v2 checkpoint SHA256:
`ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372`.
Current-best v2.2 ZIP SHA256:
`f4bcaa54157fe077318f9c0bef43d764bab3c14daa86c52a1f566213169d79c2`.
All 33 indexed frozen-v2 files and 15 indexed v2.2 release files remain unchanged.
Proof: [offline manifest](v23-offline-manifest.json) and [source proof](v23-source-proof.json).

## Submission, Git and AWS

**DEBUG: not submitted; score unavailable; delta vs 0.3331: N/A.** Local/stress
promotion failed, so the conditional single DEBUG authorization was not used.
**MAIN used this task: 0. MAIN remaining: 2/3.** Portal DEBUG used remains 1/10
from v2.2; total submission history remains six. See [quota proof](v23-quota-final.json).

Inference source commit: `551ebeb83e9c5f3003840ce2b880d52588a33f63`.
Branch: `v2.3-stability-v22-20261005`. Final report/head SHA is recorded in delivery.
Only existing CPU `i-0766a472ecb5bcf88` was started. Geometric real extraction took
37.68 seconds; stress extraction took 643.50 seconds on CPU. GPU was never started.
Zero backbone/decoder/verifier updates, optimizer/backward, new training rows,
AIHub downloads, new instances/volumes or snapshots.

EC2 API verification at **2026-10-05T03:00:26.116763+00:00** confirms **CPU stopped, GPU stopped,
retained EBS preserved**. The original encrypted 30 GiB volumes remain attached to
their original instances with **DeleteOnTermination=false**. No volume movement
or task bind mount occurred; retained results stay on the CPU root EBS. No active
TerraDelta process remained, 468 result/source files were fsynced, and filesystem
sync completed before STOP. The task SSH key was revoked with other keys preserved;
the task SSM session has no active match and the local forwarder exited.
Proof: [AWS final](v23-aws-final.json). A final idempotent STOP/API/EBS verification
follows the report push; its later timestamp is recorded in delivery.
