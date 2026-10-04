"""Fit two regularized logistic verifiers on detached feature tables only."""

import numpy as np

from terradelta.inference.verifier import FEATURES, FROZEN_V2_SHA256, validate_verifier


def fit_verifier(features, labels, valid, real, pixel_thresholds):
    x, y, valid, real = (np.asarray(v) for v in (features, labels, valid, real))
    if x.shape != (len(y), 2, len(FEATURES)) or y.shape != valid.shape or y.shape[1:] != (2,):
        raise ValueError("Invalid verifier training table")
    if real.shape != (len(y),) or real.dtype != bool or valid.dtype != bool:
        raise ValueError("Expected boolean real/valid flags")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Training features/labels must be finite and binary")
    means, scales, weights, biases, lower, upper = [], [], [], [], [], []
    for c in range(2):
        selected = valid[:, c]
        a, b, r = x[selected, c].astype(float), y[selected, c].astype(float), real[selected]
        # Equal class mass. Within positives, real/synthetic share mass equally;
        # within negatives real receives 3/4, synthetic 1/4. Missing strata are
        # renormalized; unreviewed partial-tile absence labels never participate.
        sample_weight = np.zeros(len(b))
        for positive in (False, True):
            groups = []
            for is_real, mass in ((False, .5 if positive else .25), (True, .5 if positive else .75)):
                members = (b == positive) & (r == is_real)
                if members.any():
                    groups.append((members, mass))
            if not groups:
                raise ValueError("Each verifier needs positive and negative labels")
            total = sum(mass for _, mass in groups)
            for members, mass in groups:
                sample_weight[members] = .5 * mass / total / members.sum()
        sample_weight *= len(b)
        mean = np.average(a, axis=0, weights=sample_weight)
        scale = np.maximum(np.sqrt(np.average((a - mean) ** 2, axis=0, weights=sample_weight)), 1e-3)
        design = np.column_stack([(a - mean) / scale, np.ones(len(a))])
        beta = np.zeros(design.shape[1])
        penalty = np.diag([1.] * len(FEATURES) + [0.])
        # Penalized IRLS is a convex CPU fit. No torch optimizer/backward/model
        # or any backbone/decoder parameter enters this function.
        for _ in range(100):
            logits = np.clip(design @ beta, -40, 40)
            probability = 1 / (1 + np.exp(-logits))
            gradient = design.T @ (sample_weight * (probability - b)) + penalty @ beta
            curvature = sample_weight * np.maximum(probability * (1 - probability), 1e-8)
            hessian = design.T @ (curvature[:, None] * design) + penalty + np.eye(len(beta)) * 1e-8
            step = np.linalg.solve(hessian, gradient)
            objective = np.sum(sample_weight * (np.logaddexp(0, design @ beta) - b * (design @ beta))) + .5 * beta @ penalty @ beta
            factor = 1.
            while factor > 1e-8:
                candidate = beta - factor * step
                z = design @ candidate
                loss = np.sum(sample_weight * (np.logaddexp(0, z) - b * z)) + .5 * candidate @ penalty @ candidate
                if loss <= objective:
                    break
                factor *= .5
            beta = candidate
            if np.max(np.abs(factor * step)) < 1e-8:
                break
        means.append(mean.tolist())
        scales.append(scale.tolist())
        weights.append(beta[:-1].tolist())
        biases.append(float(beta[-1]))
        lower.append(a.min(0).tolist())
        upper.append(a.max(0).tolist())
    return validate_verifier({
        "version": 2, "features": list(FEATURES), "checkpoint_sha256": FROZEN_V2_SHA256,
        "mean": means, "scale": scales, "weight": weights, "bias": biases,
        "threshold": [0., 0.], "pixel_thresholds": list(pixel_thresholds),
        "support_lower": lower, "support_upper": upper,
    })
