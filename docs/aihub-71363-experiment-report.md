# TerraDelta AIHub 71363 Experiment & Validation Report

## 1. Executive Summary

- **Experiment Name**: AIHub 71363 SkySet Hard-Negative & Object-Evidence Retraining
- **Date**: 2026-10-06 09:30 KST
- **Execution Target**: Single Bounded High-Value Training Run on CPU Instance `i-0766a472ecb5bcf88`
- **Candidate Pool**: 3,652 Existing Clean Candidates + 530 AIHub 71363 Clean Candidates = 4,182 Total Candidates
- **Outcome**: **REJECTED (Fallback to Frozen V2.3.2 Baseline Triggered)**
- **Active Production Package**: `terradelta-v232-debug.zip` (SHA256: `5aad7e163285964ce1346d97a3f8596ef630604fd9dbeee3394804b83c461788`)

---

## 2. Experimental Setup

### 2.1 Dataset Composition
- **AIHub 71363 SkySet 0.50m**: 842 total temporal pairs (75 positive new building pairs, 767 pure negative pairs).
- **Candidate Extraction**:
  - 200 positive 256x256 crops + 250 pure negative 256x256 crops.
  - Frozen Siamese ResNet18 backbone (`ca59fe...`).
  - Total candidates extracted: 626 raw components.
  - Clean retained: 530 candidates (111 TP `new_building`, 419 FP `new_building` hard negatives).
  - Enriched with 18 image-space local alignment features (`max_shift_image_px=2`).

### 2.2 Model Training
- Regularized Logistic Regression with GroupKFold cross-validation (5-fold, grouped by spatial tile).
- Investigated Schemas for `new_building`:
  - `D_align_core`: Ablation D (89 features) + 4 core alignment features (`align_residual_reduction`, `align_residual_ratio`, `align_shift_distance`, `align_cosine_gain`).
  - `D_align_reduction`: Ablation D + 1 alignment feature (`align_residual_reduction`).
  - `D_align_full`: Ablation D + 18 alignment features.
- Fixed `tree_removal`: Frozen Ablation D baseline classifier.

---

## 3. Empirical Results & Promotion Gate Evaluation

### 3.1 Validation Performance Comparison

| Variant / Baseline | Real Legacy Building Recall | Real Legacy Tree Recall | Real Legacy No-Change FP | Real Legacy Score | Stress Building TP | Stress Tree TP | Stress No-Change FP | Stress Score | Promotion Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2.3.1 Baseline** | 3/3 | 2/2 | 11/17 | 0.567203 | 113 | 97 | 0/200 | 0.691924 | PASS |
| **V2.3.2 Promoted Baseline (Hybrid)** | **3/3** | **2/2** | **8/17** | **0.628615** | **112** | **97** | **0/200** | **0.689536** | **PASS (PROMOTED)** |
| `+ AIHub 71363 (D_align_core)` | 3/3 | 2/2 | 13/17 | 0.520579 | 119 | 97 | 0/200 | 0.692176 | **FAIL (FP > 8/17)** |
| `+ AIHub 71363 (D_align_reduction)` | 3/3 | 2/2 | 14/17 | 0.520579 | 119 | 97 | 0/200 | 0.692414 | **FAIL (FP > 8/17)** |
| `+ AIHub 71363 (D_align_full)` | 1/3 | 2/2 | 12/17 | 0.389382 | 117 | 97 | 0/200 | 0.691126 | **FAIL (Recall < 3/3)** |

### 3.2 Root Cause Analysis
1. **Domain Shift in Hard Negatives**: The 419 false-positive candidates extracted from AIHub SkySet images originate from mountainous national park infrastructure (trails, rocky outcrops, observation decks, forest huts). Adding these shifted the decision boundary of the linear component classifier towards suppressing mountain features, while reducing penalty on NAIP-style suburban road/roof artifacts, leading to an increase in Real Legacy False Positives from 8/17 to 13/17.
2. **Stress Improvement vs Real Degradation**: While Stress validation true positives improved (+7 TP, reaching 119), Real Legacy score degraded from 0.628615 to 0.520579. Per competition rules, maintaining high performance on Real validation is mandatory.
3. **Strict Gate Enforcement**: The pre-established promotion gates strictly mandate `Real No-Change FP <= 8/17` and `Real Building Recall >= 3/3`. All AIHub retraining variants violated these gates.

---

## 4. Fallback Execution & Final Release Status

Per protocol:
- No unpromoted model is released or submitted.
- The system immediately fell back to the frozen **V2.3.2 Hybrid Promoted Baseline**:
  - Neural Checkpoint SHA256: `ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372`
  - Release Package: `outputs/v232/terradelta-v232-debug.zip`
  - Package SHA256: `5aad7e163285964ce1346d97a3f8596ef630604fd9dbeee3394804b83c461788`
  - Cleanroom Byte-Identical: PASS (2 passes)
  - Submitted / Verified Status: Single DEBUG submission verified on AI Factory.
  - MAIN Submission: ABSOLUTELY ZERO (reserved for final human sign-off).

---

## 5. Infrastructure Shutdown Status

- CPU Instance `i-0766a472ecb5bcf88`: **STOPPED**
- GPU Instance `i-0523a619699a95db0`: **STOPPED**
- Tunnel daemon (`task-3216`): **TERMINATED**
- No background AWS compute running.
