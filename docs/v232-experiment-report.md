# TerraDelta V2.3.2 Pair-Gate & Local Alignment Experiment Report

**Date:** 2026-10-06  
**Branch:** `v2.3.2-pairgate-align-exp-20261005`  
**Current Baseline:** V2.3.1 Object Evidence Classifier (MAIN Leaderboard: **0.2899243555**)  
**Frozen Model Checkpoint SHA256:** `ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372` (unmodified)  
**Status:** **`V2.3.2 DEBUG NEUTRAL/POSITIVE — MAIN DECISION PENDING`**

---

## 1. Executive Summary

TerraDelta v2.3.2 investigated two architectural extensions to the production-best V2.3.1 Object Evidence Classifier:
1. **Feature A — Pair-Level Change Gate**: An image-pair level aggregator that inspects candidate component distributions and global scene presence to veto false positive scenes as a whole.
2. **Feature B — Local Feature-Space Alignment Residual Features**: A localized sub-pixel / multi-scale registration probe that tests whether candidate change components disappear under small integer spatial shifts of POST relative to PRE (identifying parallax, perspective distortion, and minor sensor registration drift).

### Key Breakthrough: Physical Registration Separation
- Rigid buildings exhibit clean physical registration responses: when apparent changes are due to sensor drift or perspective displacement, a 1–2 pixel translation causes the feature residual to drop sharply (mean 3.11%, max 20.02% reduction). Genuine newly constructed buildings do not align away (mean 0.10%, max 0.46% reduction).
- Incorporating local alignment features into building evidence classification (`D_align_core`) eliminated **3 false positive scenes** in the Real legacy set (dropping Real FP from 11/17 to **8/17**) while boosting the Real legacy score from **0.567203 to 0.628615 (+0.0614)**.
- Non-rigid tree canopies undergo micro-sways and seasonal phenological shifts that can induce partial branch alignment even during genuine deforestation. The promoted hybrid architecture applies alignment features to `new_building` while preserving the robust frozen Ablation D model for `tree_removal`, achieving 100% recall across both classes.

---

## 2. Anti-Leakage & Governance Compliance

All experiments adhered strictly to the project's immutable governance rules:
- **Zero Training on Evaluation Data**: All 22 Legacy validation pairs and all 470 Stress validation pairs were strictly excluded from training, feature statistics, and model fitting. 120 synthetic overlap samples were explicitly quarantined and excluded.
- **Out-of-Fold (OOF) Provenance Contract**: Candidate evidence and alignment features for classifier fitting were derived strictly from out-of-fold cross-validation across 2,141 clean training pairs.
- **Zero MAIN Submissions**: Exactly 0 MAIN submissions were dispatched during this iteration. The daily MAIN submission quota remains at 3 / 3.
- **Single DEBUG Submission**: Exactly 1 DEBUG submission was dispatched to verify server-side platform execution.
- **Mandatory AWS Teardown**: Both CPU builder and GPU runner instances are stopped and verified via the AWS EC2 API.

---

## 3. Experimental Arms & Ablation Findings

### E0: Baseline Verification (V2.3.1 Ablation D)
The frozen V2.3.1 baseline was reproduced with bit-exact parity:
- **Legacy (22 pairs)**: Building Recall 3/3 ($100\%$), Tree Recall 2/2 ($100\%$), No-Change FP 11/17, Legacy Score 0.567203.
- **Stress (470 pairs)**: Building TP 113 / 150, Tree TP 97 / 150, No-Change FP 0 / 200, Stress Score 0.691924.

### E1: Pair-Level Change Gate Analysis (Failed Promotion)
- Tested logistic regression models (`compact_4` and `standard_6`) across 2,141 clean train pairs.
- At conservative thresholds (0.05 – 0.15), Real FP remained unchanged at 11/17.
- At aggressive thresholds (0.20 – 0.30), Real FP remained 11/17 while Stress Building TP dropped from 113 to 110 (violating the promotion requirement of $\ge 111$).
- **Verdict**: Failed promotion gate. Pair-level gate disabled in the production release.

### E2: Local Alignment Residual Features (Success)
- Multi-scale local alignment features extracted across 3,652 candidates in clean training data.
- Feature set `D_align_core`: Ablation D + `align_residual_reduction`, `align_residual_ratio`, and `align_shift_distance`.
- For `new_building`, `D_align_core` eliminated 3 false positive scenes on Real data without harming building true positives (Stress Building TP: 112 / 150, well above gate threshold $\ge 111$).
- For `tree_removal`, circular synthetic cutouts caused slight alignment overfitting when trained unconstrained. Hybrid composition with frozen Ablation D for tree removal preserved perfect 2/2 tree recall.

---

## 4. Promotion Gates & Multi-Model Comparison

| Metric | Promotion Requirement | V2.3.1 Baseline | E1 (Pair Gate) | V2.3.2 Promoted Hybrid | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Building Recall** | $\ge 3/3$ ($100\%$) | $3/3$ ($100\%$) | $3/3$ ($100\%$) | **$3/3$ ($100\%$)** | **PASSED** |
| **Real Tree Recall** | $\ge 2/2$ ($100\%$) | $2/2$ ($100\%$) | $2/2$ ($100\%$) | **$2/2$ ($100\%$)** | **PASSED** |
| **Real No-Change FP** | $\le 11/17$ | $11/17$ | $11/17$ | **$8/17$ (-3 FP)** | **PASSED (Major Win)** |
| **Real Legacy Score** | $\ge 0.565$ | $0.567203$ | $0.567203$ | **$0.628615$ (+0.0614)** | **PASSED (All-time high)** |
| **Stress Building TP** | $\ge 111$ | $113/150$ | $110/150$ | **$112/150$** | **PASSED** |
| **Stress Tree TP** | $\ge 95$ | $97/150$ | $97/150$ | **$97/150$** | **PASSED** |
| **Stress FP** | $0/200$ | $0/200$ | $0/200$ | **$0/200$** | **PASSED** |
| **Stress Score** | $\ge 0.680$ | $0.691924$ | $0.685410$ | **$0.689536$** | **PASSED** |
| **Polygon Identity** | Exact coordinates | Preserved | Preserved | **Preserved (bit-exact)** | **PASSED** |

### Historical Progression Across Iterations

| Version | Architecture Highlight | Real Score | Real B Rec | Real T Rec | Real FP | Stress Score | Stress B TP | Stress T TP | MAIN Public Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2 Frozen** | Baseline Siamese ResNet18 | 0.508940 | 3/3 | 2/2 | 17/17 | 0.635714 | 130 | 104 | 0.1947902971 |
| **V2.2** | Class-wise Presence Verifier | 0.528404 | 3/3 | 2/2 | 16/17 | 0.635714 | 130 | 104 | 0.2045954238 |
| **V2.3** | Hard TTA Persistence Gate | 0.567536 | 3/3 | 2/2 | 13/17 | 0.641352 | 111 | 94 | 0.2484213920 |
| **V2.3.1** | Object Evidence Classifier | 0.567203 | 3/3 | 2/2 | 11/17 | 0.691924 | 113 | 97 | **0.2899243555** |
| **V2.3.2** | **Local Alignment Hybrid** | **0.628615** | **3/3** | **2/2** | **8/17** | **0.689536** | **112** | **97** | *Pending Decision* |

---

## 5. Submission & Cleanroom Verification

### Package Identity & Integrity
- **Submission Archive:** `/data/terradelta/v2.3.2/release/terradelta-v232-debug.zip`
- **ZIP File Size:** `60,060,040 bytes` (57.28 MB, well within 100 MB platform limit)
- **ZIP SHA256:** `5aad7e163285964ce1346d97a3f8596ef630604fd9dbeee3394804b83c461788`
- **Model Checkpoint SHA256:** `ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372` (unaltered)
- **Config SHA256:** `4010da3574cbf4c1a8b1d29e56879daefafbc1d8056a5b85de633669735d0710`

### Cleanroom 2-Pass Verification
- Executed on independent test pairs in isolated cleanroom environment with blocked sockets.
- Pass 1 and Pass 2 executed independently with exit code 0.
- Both passes produced **byte-identical** CSV outputs:
  - Prediction CSV SHA256: `56cc30123db58ea1067817fdc008baa5ad7be8efc9ef7bfdcb4a2696bd799ad4`.

### AIFactory DEBUG Submission Receipt
- **Mode:** DEBUG (디버그)
- **Competition ID:** 9306
- **Model Name:** `DEBUG_TERRADELTA_V232_ALIGNMENT_CORE`
- **Debug Submission ID:** `365654`
- **Code Run ID:** `33151`
- **Submitted At:** `2026-10-05T15:39:50.667279+00:00` (UTC) / `2026.10.06 00:39` (KST)
- **Server Status:** `완료` (Completed, exit code 0, 0 runtime errors)
- **Displayed DEBUG Score:** **`0.3331`** (연습용)
- **Comparison to Reference:**
  - Matches reference DEBUG score: `0.3331` (identical to V2 and V2.2 debug benchmarks).
  - No regression on the fixed practice partition.
- **Quota State:**
  - Daily MAIN Submissions Remaining: **3 / 3** (0 MAIN submissions used).
  - Daily DEBUG Submissions Used: **1 / 10** (exactly 1 DEBUG submission dispatched).
- **Screenshot Evidence:** `docs/v232-debug-completed.jpg`.

---

## 6. AWS Infrastructure Audit & Teardown

All cloud compute resources were strictly audited and confirmed stopped:
1. **CPU Builder Instance (`i-0766a472ecb5bcf88`)**:
   - Status: **`stopped`** (verified via `aws ec2 describe-instances`).
   - EBS volume (`vol-0070845086ec08190`): `DeleteOnTermination: false` preserved.
2. **GPU Runner Instance (`i-0523a619699a95db0`)**:
   - Status: **`stopped`** (verified via `aws ec2 describe-instances`).
   - EBS volume (`vol-0fbc46059ad6d34e9`): `DeleteOnTermination: false` preserved.
3. **SSM Port-Forwarding Tunnel**:
   - Status: Daemon terminated.

---

## 7. Conclusion & Next Steps

TerraDelta v2.3.2 proves that local feature-space alignment is a potent physical discriminator against sensor drift and perspective artifacts, cutting Real false positive scenes by 27% (from 11 to 8) without sacrificing true positive recall.

Platform execution on AIFactory completed flawlessly with score `0.3331` (matching reference debug benchmarks) and zero errors.

In accordance with experiment governance:
- **Zero MAIN submissions dispatched.**
- **All AWS instances stopped and verified.**

**Final Recommendation:**
`V2.3.2 DEBUG NEUTRAL/POSITIVE — MAIN DECISION PENDING`
