"""Probability-to-polygon contract with per-class, independently applied settings."""

from collections.abc import Mapping

import numpy as np

from .mask import postprocess_mask
from .polygons import mask_to_polygons

CLASSES = ("new_building", "tree_removal")


def _settings(config):
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    settings = config.get("postprocess", config)
    if not isinstance(settings, Mapping):
        raise TypeError("postprocess config must be a mapping")
    return settings


def class_config(config, name):
    """Merge common config → ``classes[name]`` → direct class settings."""
    if name not in CLASSES:
        raise ValueError(f"unknown class: {name}")
    settings = _settings(config)
    aliases = {"min_component_area": "min_area", "min_positive_total_area": "min_pos_area",
               "min_total_area": "min_pos_area", "simplification_tolerance": "simplify_px"}

    def canonicalize(layer):
        normalized = dict(layer)
        for alias, canonical in aliases.items():
            if alias in normalized:
                normalized.setdefault(canonical, normalized[alias])
                normalized.pop(alias)
        morphology = {key: layer[key] for key in ("opening", "closing", "dilation", "erosion", "fill_holes")
                      if key in layer}
        morphology.update(layer.get("morphology") or {})
        if morphology:
            normalized["morphology"] = morphology
        return normalized

    result = canonicalize({key: value for key, value in settings.items()
                           if key not in (*CLASSES, "classes", "mode")})
    classes = settings.get("classes") or {}
    if not isinstance(classes, Mapping):
        raise TypeError("postprocess.classes must be a mapping")
    for override in (classes.get(name) or {}, settings.get(name) or {}):
        if not isinstance(override, Mapping):
            raise TypeError(f"{name} config must be a mapping")
        override = canonicalize(override)
        inherited_morphology = dict(result.get("morphology") or {})
        inherited_morphology.update(override.get("morphology") or {})
        result.update(override)
        if inherited_morphology:
            result["morphology"] = inherited_morphology
    threshold_key = "building_threshold" if name == "new_building" else "tree_threshold"
    if "threshold" not in result:
        result["threshold"] = result.get(threshold_key, .5)
    return result


def probabilities_to_masks(probabilities, config=None):
    """Threshold-eligible positives compete with background by probability.

    Background wins ties; building wins positive ties (argmax channel order).
    Morphology conflicts are resolved by the original positive probabilities.
    Argmax behavior, including its optional morphology, remains unchanged.
    """
    p = np.asarray(probabilities)
    if p.ndim != 3 or p.shape[0] != 3 or not all(p.shape[1:]):
        raise ValueError("probabilities must have shape (3, H, W) with H,W > 0")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and within [0, 1]")
    mode = _settings(config).get("mode", "argmax")
    if mode not in ("argmax", "threshold", "probability"):
        raise ValueError("mode must be 'argmax' or 'threshold'")
    options_by_class = {name: class_config(config, name) for name in CLASSES}
    for name, options in options_by_class.items():
        threshold = options["threshold"]
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{name} threshold must be within [0, 1]")
    if mode == "argmax":
        labels = p.argmax(axis=0)
    else:
        eligible = p.astype(np.float64, copy=True)
        for channel, name in enumerate(CLASSES, 1):
            eligible[channel] = np.where(p[channel] >= options_by_class[name]["threshold"], p[channel], -1)
        labels = eligible.argmax(axis=0)
    output = {}
    for channel, name in enumerate(CLASSES, start=1):
        options = options_by_class[name]
        mask = labels == channel
        output[name] = postprocess_mask(mask, options)
    if mode != "argmax":
        overlap = output[CLASSES[0]] & output[CLASSES[1]]
        output[CLASSES[0]][overlap & (p[2] > p[1])] = False
        output[CLASSES[1]][overlap & (p[1] >= p[2])] = False
    return output


def predictions_to_polygons(probabilities, config=None):
    """CHW probabilities → {'new_building': exteriors, 'tree_removal': exteriors}.

    Empty config defaults to argmax and the exact reference backend. No
    morphology, TTA, or polygon cap is implicitly enabled.
    """
    masks = probabilities_to_masks(probabilities, config)
    output = {}
    for name in CLASSES:
        options = class_config(config, name)
        output[name] = mask_to_polygons(
            masks[name], min_area=options.get("min_area", 30),
            min_pos_area=options.get("min_pos_area", 20),
            simplify_px=options.get("simplify_px", .5), ndigits=options.get("ndigits", 2),
            backend=options.get("backend", "reference"), max_polygons=options.get("max_polygons"))
    return output
