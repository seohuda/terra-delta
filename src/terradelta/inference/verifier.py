"""Class-wise linear change verification; never modifies frozen pixel predictions."""

import hashlib
from pathlib import Path

import numpy as np

FROZEN_V2_SHA256 = "ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372"
FEATURES = (
    "presence", "prob_mean", "prob_std", "prob_max", "prob_q90", "prob_q99", "mask_fraction",
    "global_rgb_difference", "roi_rgb_difference", "relative_rgb_difference",
    "global_corrected_difference", "roi_corrected_difference", "relative_corrected_difference",
    "roi_luminance_before", "roi_luminance_after", "roi_luminance_change",
    "roi_green_before", "roi_green_after", "roi_green_loss", "roi_edge_before", "roi_edge_after",
    "roi_edge_change", "global_edge_change", "roi_corrected_q90",
)


def verify_frozen_checkpoint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != FROZEN_V2_SHA256:
        raise ValueError("Verifier requires the exact frozen v2 checkpoint")


def change_features(images, pixels, presence, pixel_thresholds):
    """Pool fixed image/probability statistics in each frozen class candidate mask.

    Inputs use the exact v2 RGB normalization. Global affine color correction
    provides an appearance-normalized residual, without resampling either image.
    Empty candidate masks yield zero ROI statistics, rather than NaNs.
    """
    images, pixels, presence = (np.asarray(v, dtype=np.float64) for v in (images, pixels, presence))
    n = len(images)
    if images.shape != (n, 6, 256, 256) or pixels.shape != (n, 2, 256, 256) or presence.shape != (n, 2):
        raise ValueError("Incompatible verifier input shapes")
    if not all(np.isfinite(v).all() for v in (images, pixels, presence)):
        raise ValueError("Verifier inputs must be finite")
    if any(((v < 0) | (v > 1)).any() for v in (pixels, presence)):
        raise ValueError("Verifier probabilities must be in [0,1]")
    thresholds = np.asarray(pixel_thresholds, dtype=float)
    if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Expected two finite pixel thresholds in [0,1]")
    rgb = images.reshape(n, 2, 3, 256, 256)
    rgb = np.clip(rgb * np.array([.229, .224, .225])[None, None, :, None, None]
                  + np.array([.485, .456, .406])[None, None, :, None, None], 0, 1)
    result = np.empty((n, 2, len(FEATURES)), dtype=np.float64)
    for i, pair in enumerate(rgb):
        before, after = pair
        difference = np.abs(after - before).mean(0)
        bmean, amean = before.mean((1, 2), keepdims=True), after.mean((1, 2), keepdims=True)
        ratio = np.clip(after.std((1, 2), keepdims=True) / np.maximum(before.std((1, 2), keepdims=True), .02), .5, 2)
        corrected = np.abs(after - np.clip((before - bmean) * ratio + amean, 0, 1)).mean(0)
        luminance = pair.mean(1)
        green = 2 * pair[:, 1] - pair[:, 0] - pair[:, 2]
        dy, dx = np.gradient(luminance, axis=(1, 2))
        edges = np.hypot(dx, dy)
        edge_change = np.abs(edges[1] - edges[0])
        for c in range(2):
            p = pixels[i, c]
            mask = p >= thresholds[c]
            def pool(value):
                return float(value[mask].mean()) if mask.any() else 0.0
            d, r = pool(difference), pool(corrected)
            result[i, c] = [
                presence[i, c], p.mean(), p.std(), p.max(), *np.quantile(p, [.9, .99]), mask.mean(),
                difference.mean(), d, d / max(difference.mean(), .01),
                corrected.mean(), r, r / max(corrected.mean(), .01),
                pool(luminance[0]), pool(luminance[1]), pool(luminance[1] - luminance[0]),
                pool(green[0]), pool(green[1]), pool(green[0] - green[1]),
                pool(edges[0]), pool(edges[1]), pool(edge_change), edge_change.mean(),
                float(np.quantile(corrected[mask], .9)) if mask.any() else 0.0,
            ]
    return result


def validate_verifier(config):
    if not isinstance(config, dict) or config.get("version") not in (1, 2) or config.get("features") != list(FEATURES):
        raise ValueError("Unsupported verifier feature contract")
    if config.get("checkpoint_sha256") != FROZEN_V2_SHA256:
        raise ValueError("Verifier checkpoint identity differs")
    width = len(FEATURES)
    shapes = [("mean", (2, width)), ("scale", (2, width)), ("weight", (2, width)),
              ("bias", (2,)), ("threshold", (2,)), ("pixel_thresholds", (2,))]
    if config["version"] == 2:
        shapes.extend((key, (2, width)) for key in ("support_lower", "support_upper"))
    for key, shape in shapes:
        try:
            values = np.asarray(config[key], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid verifier {key}") from error
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid verifier {key}")
        if key == "scale" and (values <= 0).any():
            raise ValueError("Verifier scale must be positive")
        if key in ("threshold", "pixel_thresholds") and ((values < 0) | (values > 1)).any():
            raise ValueError(f"Verifier {key} must be in [0,1]")
    if config["version"] == 2 and (np.asarray(config["support_lower"]) > np.asarray(config["support_upper"])).any():
        raise ValueError("Verifier support bounds are reversed")
    return config


def verifier_scores(features, config):
    config = validate_verifier(config)
    x = np.asarray(features, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (2, len(FEATURES)) or not np.isfinite(x).all():
        raise ValueError("Invalid verifier features")
    logits = (((x - np.asarray(config["mean"])) / np.asarray(config["scale"]))
              * np.asarray(config["weight"])).sum(2) + np.asarray(config["bias"])
    scores = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
    if config["version"] == 2:
        # A rejection model has no evidence outside its observed training support.
        # Abstain there by retaining the original frozen-v2 class prediction.
        outside = ((x < np.asarray(config["support_lower"])) | (x > np.asarray(config["support_upper"]))).any(2)
        scores = np.where(outside, 1., scores)
    return scores
