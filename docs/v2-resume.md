# Completed-run v2 evaluation

This run reuses `siamese-align-v1` checkpoints and existing manifests. It generates
no data and executes no new training. The trained architecture remains unchanged:
shared ResNet18, independent sigmoid segmentation and presence heads, and learned
POST-to-PRE feature alignment at scales 3 and 4. Deployment normalization uses
float32 RGB operations matching `IndependentChangeDataset`.

The prepared training manifest has 2,261 rows. The saved completed-run summary
records **2,200 actual synthetic training samples**, with 700 building, 700 tree,
100 both and 700 negatives. The prepared real negatives were not used in that
training run. Real legacy validation contains 22 WA samples; synthetic WA stress
contains 470 samples. Keep their results separate; neither set is an independent
post-selection estimate, since thresholds are chosen on those same sets.

`calibrate_v2.py` compares only steps 50, 150 and 250. Each class searches ten
presence thresholds from 0 to 0.5, presence disabled, and eight pixel thresholds
from 0.35 to 0.70. Reference polygonization uses min_area=30, min_pos_area=20,
simplify_px=0.5 and ndigits=2. Per-class utility gives the real and synthetic set
equal weight. Within 0.01 of the maximum it favors the best immediate-neighbor
minimum, then neighbor mean. Checkpoint selection averages those class utilities.
Caches and completed results carry checkpoint, manifest and calibration-code
hashes. Prediction masks are independent and may overlap.

`finalize_v2.py` repeats actual selected inference on both sets, verifies exported
polygon metrics against the sweep, writes both prediction CSVs and
`/data/terradelta/v2/evaluations/final-v2-selection.json`, and creates a reproducible
ZIP on retained EBS. The established submission entrypoint is root
`predict.ipynb`; config, source and weights are bundled under `assets/`. Inference
requirements exclude training, data generation and the submission SDK.

`verify_v2_zip.py` runs the notebook twice in isolated Python using only extracted
ZIP modules. It blocks sockets, weight downloads, optimizer construction and
backward, forces CPU fallback, checks real and synthetic image pairs, empty cells,
ID preservation, schema, valid finite bounded polygons, independent overlapping
heads and presence gating before pixel conversion. Outputs must match byte for
byte. Unit tests also verify a real v2 checkpoint/export/notebook path.

Existing DEBUG references were retrieved without resubmission: organizer original
364433 / run 31003 scored 0.2990; old balanced 364413 / run 30967 scored 0.1458.
New v2 submission is restricted to one DEBUG. Main submission is prohibited. The
current submission UI exposes status and practice score, but not detailed server
logs, runtime or prediction CSV.

During evaluation an external AWS operation terminated the original GPU and
reattached preserved 30 GiB EBS to the existing CPU builder. The user explicitly
approved finishing on that builder. No replacement GPU was created. CPU shutdown
and EBS retention are verified after completion. Final metrics, DEBUG registration,
artifact hashes and AWS proof are saved in the accompanying JSON report.

Final selected step is 250. Building gate is disabled, pixel threshold 0.35;
tree presence threshold is 0.30 and pixel threshold 0.70. Actual re-inference
exactly matches cached polygon scores within 1e-10: real legacy 0.5089401836663427
and synthetic stress 0.6357135502609976. Real no-change FP is 17/17; synthetic
no-change FP is 42/200. Both pixel choices reach a searched grid boundary, and
the strongest building neighbor drops equal-set utility from 0.5464 to 0.4838.
This limits claims of calibration robustness. No further threshold tuning will
use the DEBUG score. The original notice files were reused from the known working
baseline package because its original source directory was absent on restored EBS.

The full local suite passed 493 tests. Two focused exporter regressions also
passed after correcting v2 handling of training-format metadata. Ruff includes
the repository notebooks; E402 is specifically exempted in the three existing
notebooks that intentionally establish their import path before local imports.

One v2 DEBUG completed: submission **364583**, run **31278**, displayed practice
score **0.3331**. This is **+0.0341** against original baseline 0.2990 and
**+0.1873** against old balanced 0.1458. Main quota remains 3/3; new DEBUG count
is one, new Main count zero. Server runtime, detailed logs, warnings and exact
CSV remain unavailable in the current UI. The measured DEBUG score improves,
but the real no-change FP result remains a material risk before Main approval.

**V2 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL**

Final AWS API proof at 2026-10-04 12:55:19 UTC: CPU builder stopped; original
GPU terminated externally, so it cannot be stopped/restarted. Encrypted 30 GiB
volume vol-0070845086ec08190 remains attached to the stopped CPU builder with
DeleteOnTermination=false. The task SSH authorization was revoked and its SSM
forwarding session terminated. Checkpoints, data, full sweep, actual prediction
CSVs, selected config, source, frozen ZIP and DEBUG registration/result manifest
remain on that EBS. The post-stop AWS proof is saved in this repository.
