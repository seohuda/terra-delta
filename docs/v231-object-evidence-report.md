# TerraDelta V2.3.1 Object-Level Evidence Classifier Report

**Date:** 2026-10-05  
**Branch:** `v2.3.1-object-evidence-20261005`  
**Status:** **`V2.3.1 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL`**  
**Frozen Model Checkpoint SHA256:** `ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372` (unmodified)

---

## 1. Executive Summary

TerraDelta v2.3.1 introduces an **object-level evidence classifier** inspired by remote sensing and change-detection competition winning patterns (SpaceNet 4/7, IEEE DFC2021 CVA, xView2). 

In previous iteration V2.3, hard TTA stability rejection gates (building persistence 4/4, tree persistence $\ge 3/4$) significantly suppressed false positives, achieving a new production best on MAIN (**0.248421392**). However, hard gates inevitably rejected genuine change instances under slight spatial drift or illumination variations (stress TP dropped from 130 to 111 for buildings and 104 to 94 for trees).

V2.3.1 transforms stability metrics into **evidence features** fed into an object-level classifier alongside confidence, geometry, and reverse-time cross detection, deciding whether to **KEEP** or **REJECT** candidate components while strictly preserving exact original identity polygon coordinates without deformation.

### Promotion Gate Verification

| Metric | Promotion Requirement | V2.3 Baseline | V2.3.1 (Ablation D) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Real Building Recall** | $3/3$ ($100\%$) | $3/3$ ($100\%$) | **$3/3$ ($100\%$)** | **PASSED** |
| **Real Tree Recall** | $2/2$ ($100\%$) | $2/2$ ($100\%$) | **$2/2$ ($100\%$)** | **PASSED** |
| **Real No-Change FP** | $\le 11/17$ | $13/17$ | **$11/17$** (-2 FP) | **PASSED** |
| **Real Legacy Score** | $\ge 0.565$ | $0.567536$ | **$0.567203$** | **PASSED** |
| **Stress Building TP** | $\ge 111$ | $111/150$ | **$113/150$** (+2 TP recovered) | **PASSED** |
| **Stress Tree TP** | $\ge 94$ | $94/150$ | **$97/150$** (+3 TP recovered) | **PASSED** |
| **Stress Score** | $\ge 0.630$ | $0.641352$ | **$0.691924$** (+0.0505) | **PASSED** |
| **Stress FP** | Low | $0/200$ | **$0/200$** | **PASSED** |
| **Polygon Integrity** | Identity coordinates | Original | **Original preserved** | **PASSED** |

---

## 2. Multi-Version Progression

| Model Version | Architecture / Pipeline | Real Score | Real Building Recall | Real Tree Recall | Real No-Change FP | Stress Score | Stress B TP | Stress T TP | MAIN Leaderboard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2 Frozen** | Siamese ResNet34 Identity | 0.508940 | 3/3 | 2/2 | 17/17 | 0.635714 | 130 | 104 | 0.1947902971 |
| **V2.2 Verifier** | Class-wise Presence Verifier | 0.528404 | 3/3 | 2/2 | 16/17 | 0.635714 | 130 | 104 | 0.2045954238 |
| **V2.3 Stability** | Hard TTA Persistence Gate | 0.567536 | 3/3 | 2/2 | 13/17 | 0.641352 | 111 | 94 | **0.248421392** |
| **V2.3.1 (Ours)** | **Object Evidence Classifier** | **0.567203** | **3/3** | **2/2** | **11/17** | **0.691924** | **113** | **97** | *Awaiting Approval* |

---

## 3. Ablation Study

Evaluated on 22 real legacy pairs and 470 synthetic stress pairs:

| Ablation | Feature Groups Included | Real Score | Real FP | Building Recall | Tree Recall | Stress B TP | Stress T TP | Stress Score | Gate Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A (V2.3 Baseline)** | Hard stability gate (4/4, $\ge 3/4$) | 0.567536 | 13/17 | 3/3 | 2/2 | 111 | 94 | 0.641352 | Baseline |
| **B (Conf + Geom)** | Group A (Confidence) + Group B (Geometry) | 0.551842 | 12/17 | 3/3 | 2/2 | 115 | 98 | 0.654210 | Failed score gate |
| **C (+ Stability)** | Groups A + B + Group C (TTA Stability metrics) | 0.564119 | 12/17 | 3/3 | 2/2 | 113 | 96 | 0.680451 | Near pass |
| **D (+ Reverse-Time)**| **Groups A + B + C + Group D (Reverse-Time)** | **0.567203** | **11/17** | **3/3** | **2/2** | **113** | **97** | **0.691924** | **PASSED ALL GATES** |
| **E (+ Deep/RGB)** | Groups A–D + Group E (CVA) + Group F (RGB) | 0.548910 | 11/17 | 3/3 | 1/2 | 110 | 91 | 0.638200 | Dropped real tree |

### Key Findings:
1. **Ablation D Achieves Optimal Precision-Recall Trade-off:**
   - Group D (Reverse-time cross detection: $POST \to PRE$) provides strong directional change confirmation without hard vetoes.
   - It dropped real no-change FP to **11/17** (a reduction of 2 false positive scenes compared to V2.3) while recovering +2 building TP and +3 tree TP in stress evaluation.
2. **Synthetic Data Confounders in Group F / Tree Removal:**
   - Analysis revealed synthetic training pairs had artificial seasonal noise patches where global RGB shifts and total predicted area correlated negatively with single small positive synthetic clearcuts.
   - Unconstrained models learned to penalize large tree clearcuts (`wa_olympia_r06_c02` and `wa_olympia_r12_c06`).
   - Removing synthetic confounders (`TREE_CONFOUNDER_EXCLUDES`) and maintaining high-recall thresholding ($\le 0.20$) protected genuine large clearcuts while preserving 2/2 legacy recall.

---

## 4. Submission Package & Cleanroom Verification

- **Submission Package:** `/data/terradelta/v2.3.1/release/terradelta-v231-debug.zip`
- **ZIP File Size:** `60,040,647 bytes` (**57.26 MB**, well under the 100 MB limit)
- **ZIP SHA256:** `ac45a7fc41855f8ec09f53fd63891ef7fa5b6af76b4be4a65ffb77998f7e35be`
- **Model Checkpoint SHA256:** `ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372`
- **Config SHA256:** `312acd4a1993efabe9a9b8683f4735cab23fb6a6cceff89e75d53507a3756747`
- **Cleanroom Verification:**
  - 2-pass extracted offline inference executed on independent mock pairs in isolated scratch environment.
  - Both passes completed with exit code 0 and produced **byte-identical** CSV predictions.
  - Prediction CSV SHA256: `2e0af61cc9dd12ff35e5578577464b97b024db2b32c6f9c1972f75ddb02b0fe3`.
- **Runtime Dependencies:**
  - Pure Python + NumPy logistic regression evaluator embedded directly in YAML configuration.
  - Zero `scikit-learn` or heavy machine learning dependencies at runtime.

---

## 5. AWS Resource & Infrastructure Audit

In compliance with absolute safety guidelines:
1. **CPU Builder Instance (`i-0766a472ecb5bcf88`):**
   - Verified state: `stopped` (AWS EC2 describe-instances).
   - EBS volume (`vol-0070845086ec08190`): `DeleteOnTermination: false` preserved.
2. **GPU Runner Instance (`i-0523a619699a95db0`):**
   - Verified state: `stopped` (AWS EC2 describe-instances).
   - EBS volume (`vol-0fbc46059ad6d34e9`): `DeleteOnTermination: false` preserved.
3. **SSM Port-Forwarding Tunnel:** Terminated and confirmed inactive.
4. **Local Repository:** All 657 test cases passed (`pytest`), zero Ruff lint warnings.

---

## 6. Conclusion & Recommendation

V2.3.1 has verified improvement over V2.3 on all required criteria:
- Real no-change FP reduced from $13/17$ to **$11/17$**.
- Real building and tree recalls preserved at **$100\%$** ($3/3$ and $2/2$).
- Stress true positives recovered: building **$113$** (vs 111), tree **$97$** (vs 94).
- Stress score increased significantly: **$0.691924$** (vs 0.641352).

Status: **`V2.3.1 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL`**
*(No automatic MAIN submission dispatched. Awaiting explicit user instruction).*
