# TerraDelta Lessons Learned

## 2026-10-05: V2.3.2 Code Review Hardening Patterns

1. **Physical/Image Space Units for Spatial Budgets**:
   - Never configure spatial search ranges (e.g. shifts, padding) in feature-map pixels when dealing with multi-scale feature pyramids.
   - Always define the public budget in canonical image pixels (e.g. `max_shift_image_px`), and derive integer feature-map shifts per scale deterministically using stride.

2. **Fair Support in Shift/Registration Comparisons**:
   - Shifting an ROI near image borders can push pixels out of bounds. If residuals are averaged over whatever pixels remain, a shift can artificially win simply by discarding mismatched border pixels.
   - Always compare candidate shifts on a shared common support region (or enforce strict overlap requirements and evaluate baseline on the exact same support).

3. **Zero-Residual and Edge Case Semantics**:
   - When baseline difference `residual_before <= EPS`, naive formulas like `residual_after / max(residual_before, EPS)` yield `0.0`, which falsely indicates 100% reduction.
   - Explicitly define: if `residual_before <= EPS`, `residual_ratio = 1.0` and `residual_reduction = 0.0`.

4. **Model Binding to Feature Extraction Fingerprint**:
   - Feature definitions can subtly change between training and inference (e.g. scales, thresholds, window sizes).
   - Serialized models must record the feature extraction schema/metadata fingerprint, and inference must fail fast on mismatch.

5. **Cache Invalidation Provenance**:
   - Caching spatial features requires identity of the source imagery and model weights (fingerprints), not just the bounding box or config.
   - When persistent caching is requested, require caller-provided model and input fingerprints. Validate that cached feature values are finite.

6. **Eliminate Silent No-Ops in Pipeline Configurations**:
   - If an experimental flag (such as `pair_gate.enabled=true`) depends on features extracted in an earlier stage, do not let disabling an earlier stage (such as `classifier.enabled=false`) silently bypass the experimental stage.
   - Provide a clean fallback (e.g., confidence fallback) or reject the configuration explicitly.

7. **Reuse Shared Encoder Representations**:
   - When multiple feature extractors (e.g. deep CVA and local alignment) require frozen encoder representations, compute the encoder pyramid once and retain only the requested scale slices on CPU, freeing GPU tensors immediately.
