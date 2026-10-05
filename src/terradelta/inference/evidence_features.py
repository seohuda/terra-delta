"""Object-level evidence feature extraction for candidate change components.

Extracts multi-source evidence per candidate component:
- Group A: Model confidence & presence
- Group B: Geometry, morphology & neighborhood context
- Group C: Multi-view TTA stability
- Group D: Cross-detection / reverse-time asymmetry
- Group E: Deep Siamese Change Vector Analysis (CVA)
- Group F: RGB / edge / texture structural evidence & local-ring contrast
- Class-specific proxies (building edge density, tree vegetation loss)
- Global image-pair nuisance shifts

Candidate polygons and identity component masks remain the sole shape source.
No polygon deformation, dilation, erosion or smoothing is applied.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.ndimage as ndi
from shapely.geometry import Polygon
import torch
import torch.nn.functional as F

from .stability_v23 import match_components

# Standard feature order for deterministic serialization and model training
BASE_CONFIDENCE_FEATURES = (
    "component_area",
    "mean_probability",
    "max_probability",
    "median_probability",
    "p90_probability",
    "p95_probability",
    "top1_percent_mean",
    "top5_percent_mean",
    "confidence_std",
    "presence_score",
    "verifier_score",
    "total_class_area",
    "num_same_class_components",
)

BASE_GEOMETRY_FEATURES = (
    "area",
    "perimeter",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "min_area_rect_width",
    "min_area_rect_height",
    "min_area_rect_ratio",
    "convex_hull_area",
    "solidity",
    "extent",
    "eccentricity",
    "circularity",
    "rectangularity",
    "major_axis",
    "minor_axis",
    "border_distance",
    "touches_border",
    "long_thin_score",
    "nearest_component_distance",
    "nearby_component_area",
    "area_over_median_neighbor",
)

STABILITY_FEATURES = (
    "persistence_count",
    "persistence_fraction",
    "mean_matching_iou",
    "minimum_matching_iou",
    "maximum_matching_iou",
    "matching_area_mean",
    "area_std",
    "area_cv",
    "centroid_mean_drift",
    "centroid_max_drift",
    "tta_prob_mean",
    "tta_prob_std",
)

REVERSE_FEATURES = (
    "reverse_mean_prob",
    "reverse_max_prob",
    "reverse_overlap_fraction",
    "reverse_matching_iou",
    "prob_diff_forward_reverse",
    "area_ratio_forward_reverse",
    "reverse_asymmetry",
)

DEEP_CVA_FEATURES = (
    "cva_mean",
    "cva_max",
    "cva_std",
    "cva_p90",
    "cva_ring_mean",
    "cva_ratio",
    "cva_diff",
    "cva_cosine_dist",
)

RGB_STRUCTURAL_FEATURES = (
    "rgb_diff_mean",
    "rgb_diff_r",
    "rgb_diff_g",
    "rgb_diff_b",
    "luminance_diff",
    "local_variance_diff",
    "sobel_mag_diff",
    "edge_density_pre",
    "edge_density_post",
    "edge_density_delta",
    "local_hist_dist",
    "ring_rgb_diff_mean",
    "rgb_ratio",
    "rgb_diff_ring_diff",
    "sobel_ratio",
)

GLOBAL_NUISANCE_FEATURES = (
    "global_rgb_shift",
    "global_brightness_shift",
    "global_contrast_shift",
    "global_edge_diff",
    "local_vs_global_rgb",
    "local_vs_global_edge",
)

BUILDING_SPECIFIC_FEATURES = (
    "bld_post_edge_density",
    "bld_pre_edge_density",
    "bld_edge_increase",
    "bld_structural_change",
    "bld_rectangularity",
    "bld_solidity",
)

TREE_SPECIFIC_FEATURES = (
    "tree_exg_pre",
    "tree_exg_post",
    "tree_exg_drop",
    "tree_green_ratio_pre",
    "tree_green_ratio_post",
    "tree_green_ratio_drop",
    "tree_texture_change",
)

# Ordered feature schemas for each ablation
ABLATION_SCHEMAS = {
    "B": BASE_CONFIDENCE_FEATURES + BASE_GEOMETRY_FEATURES,
    "C": BASE_CONFIDENCE_FEATURES + BASE_GEOMETRY_FEATURES + STABILITY_FEATURES,
    "D": BASE_CONFIDENCE_FEATURES + BASE_GEOMETRY_FEATURES + STABILITY_FEATURES + REVERSE_FEATURES,
    "E_building": (
        BASE_CONFIDENCE_FEATURES
        + BASE_GEOMETRY_FEATURES
        + STABILITY_FEATURES
        + REVERSE_FEATURES
        + DEEP_CVA_FEATURES
        + RGB_STRUCTURAL_FEATURES
        + GLOBAL_NUISANCE_FEATURES
        + BUILDING_SPECIFIC_FEATURES
    ),
    "E_tree": (
        BASE_CONFIDENCE_FEATURES
        + BASE_GEOMETRY_FEATURES
        + STABILITY_FEATURES
        + REVERSE_FEATURES
        + DEEP_CVA_FEATURES
        + RGB_STRUCTURAL_FEATURES
        + GLOBAL_NUISANCE_FEATURES
        + TREE_SPECIFIC_FEATURES
    ),
}
ABLATION_SCHEMAS["E_new_building"] = ABLATION_SCHEMAS["E_building"]
ABLATION_SCHEMAS["E_tree_removal"] = ABLATION_SCHEMAS["E_tree"]


def get_component_pixels(part: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Get integer pixel coordinates (y, x) inside a shapely polygon."""
    from shapely import contains_xy

    xmin, ymin, xmax, ymax = (int(v) for v in part.bounds)
    xmin = max(0, min(255, xmin))
    ymin = max(0, min(255, ymin))
    xmax = max(0, min(256, xmax))
    ymax = max(0, min(256, ymax))
    if xmax <= xmin or ymax <= ymin:
        return np.array([], dtype=int), np.array([], dtype=int)
    y, x = np.mgrid[ymin:ymax, xmin:xmax]
    inside = contains_xy(part, x + 0.5, y + 0.5)
    return y[inside], x[inside]


def get_surrounding_ring_mask(y: np.ndarray, x: np.ndarray, radius: int = 5, shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a dilated ring mask: dilate(component, r) - component."""
    mask = np.zeros(shape, dtype=bool)
    if len(y) > 0:
        mask[y, x] = True
    dilated = ndi.binary_dilation(mask, iterations=radius)
    return dilated & (~mask)


def compute_geometry_features(
    poly: Polygon,
    y: np.ndarray,
    x: np.ndarray,
    all_polys: Sequence[Polygon],
    current_idx: int,
    image_size: tuple[int, int] = (256, 256),
) -> dict[str, float]:
    """Extract geometry, morphology and neighborhood context features."""
    area = float(poly.area)
    perimeter = float(poly.length)
    minx, miny, maxx, maxy = poly.bounds
    w = max(0.0, float(maxx - minx))
    h = max(0.0, float(maxy - miny))
    aspect_ratio = max(w, h) / max(min(w, h), 1e-4)

    # Minimum area rectangle (oriented bbox)
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords) if hasattr(rect, "exterior") else []
    if len(coords) >= 4:
        p0, p1, p2 = np.array(coords[0]), np.array(coords[1]), np.array(coords[2])
        side1 = float(np.linalg.norm(p1 - p0))
        side2 = float(np.linalg.norm(p2 - p1))
        min_rect_w = min(side1, side2)
        min_rect_h = max(side1, side2)
    else:
        min_rect_w = min(w, h)
        min_rect_h = max(w, h)
    min_rect_ratio = min_rect_h / max(min_rect_w, 1e-4)
    min_rect_area = min_rect_w * min_rect_h

    # Convex hull & solidity
    ch_area = float(poly.convex_hull.area)
    solidity = area / max(ch_area, 1e-4)
    extent = area / max(w * h, 1e-4)
    rectangularity = area / max(min_rect_area, 1e-4)
    circularity = 4.0 * math.pi * area / max(perimeter**2, 1e-4)

    # Moments & eccentricity
    if len(x) >= 2:
        cx, cy = float(np.mean(x)), float(np.mean(y))
        dx = x - cx
        dy = y - cy
        mu20 = float(np.mean(dx * dx))
        mu02 = float(np.mean(dy * dy))
        mu11 = float(np.mean(dx * dy))
        term = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * mu11 * mu11))
        lam1 = (mu20 + mu02 + term) / 2.0
        lam2 = max(0.0, (mu20 + mu02 - term) / 2.0)
        eccentricity = math.sqrt(max(0.0, 1.0 - lam2 / max(lam1, 1e-6)))
        major_axis = 4.0 * math.sqrt(max(0.0, lam1))
        minor_axis = 4.0 * math.sqrt(max(0.0, lam2))
    else:
        eccentricity = 0.0
        major_axis = max(w, h)
        minor_axis = min(w, h)

    # Border distances
    dist_left = minx
    dist_top = miny
    dist_right = image_size[1] - maxx
    dist_bottom = image_size[0] - maxy
    border_distance = max(0.0, float(min(dist_left, dist_top, dist_right, dist_bottom)))
    touches_border = 1.0 if border_distance <= 0.5 else 0.0
    long_thin_score = perimeter / (2.0 * math.sqrt(math.pi * max(area, 1e-4)))

    # Context features with respect to other same-class components
    other_polys = [p for j, p in enumerate(all_polys) if j != current_idx]
    if other_polys:
        centroid = poly.centroid
        distances = [centroid.distance(o.centroid) for o in other_polys]
        nearest_component_distance = float(min(distances))
        nearby_component_area = float(sum(o.area for o, d in zip(other_polys, distances) if d <= 50.0))
        median_neighbor_area = float(np.median([o.area for o in other_polys]))
        area_over_median_neighbor = area / max(median_neighbor_area, 1e-4)
    else:
        nearest_component_distance = float(image_size[0])
        nearby_component_area = 0.0
        area_over_median_neighbor = 1.0

    return {
        "area": area,
        "perimeter": perimeter,
        "bbox_width": w,
        "bbox_height": h,
        "aspect_ratio": aspect_ratio,
        "min_area_rect_width": min_rect_w,
        "min_area_rect_height": min_rect_h,
        "min_area_rect_ratio": min_rect_ratio,
        "convex_hull_area": ch_area,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
        "circularity": circularity,
        "rectangularity": rectangularity,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "border_distance": border_distance,
        "touches_border": touches_border,
        "long_thin_score": long_thin_score,
        "nearest_component_distance": nearest_component_distance,
        "nearby_component_area": nearby_component_area,
        "area_over_median_neighbor": area_over_median_neighbor,
    }


def compute_confidence_features(
    y: np.ndarray,
    x: np.ndarray,
    prob_map: np.ndarray,
    presence_score: float,
    verifier_score: float,
    all_areas: Sequence[float],
) -> dict[str, float]:
    """Extract model confidence and presence features."""
    if len(y) > 0:
        values = prob_map[y, x]
        mean_p = float(np.mean(values))
        max_p = float(np.max(values))
        med_p = float(np.median(values))
        p90 = float(np.quantile(values, 0.90))
        p95 = float(np.quantile(values, 0.95))
        n_top1 = max(1, int(math.ceil(0.01 * len(values))))
        n_top5 = max(1, int(math.ceil(0.05 * len(values))))
        sorted_vals = np.sort(values)
        top1 = float(np.mean(sorted_vals[-n_top1:]))
        top5 = float(np.mean(sorted_vals[-n_top5:]))
        std_p = float(np.std(values))
        comp_area = float(len(values))
    else:
        mean_p = max_p = med_p = p90 = p95 = top1 = top5 = std_p = comp_area = 0.0

    return {
        "component_area": comp_area,
        "mean_probability": mean_p,
        "max_probability": max_p,
        "median_probability": med_p,
        "p90_probability": p90,
        "p95_probability": p95,
        "top1_percent_mean": top1,
        "top5_percent_mean": top5,
        "confidence_std": std_p,
        "presence_score": float(presence_score),
        "verifier_score": float(verifier_score),
        "total_class_area": float(sum(all_areas)),
        "num_same_class_components": float(len(all_areas)),
    }


def compute_reverse_features(
    y: np.ndarray,
    x: np.ndarray,
    poly: Polygon,
    reverse_prob: np.ndarray,
    reverse_presence: float,
    reverse_components: Sequence[Polygon],
    pixel_threshold: float = 0.5,
) -> dict[str, float]:
    """Extract reverse time cross-detection features.

    Compares forward prediction with reverse-time prediction (POST->PRE).
    """
    if len(y) > 0:
        rev_vals = reverse_prob[y, x]
        rev_mean = float(np.mean(rev_vals))
        rev_max = float(np.max(rev_vals))
        rev_overlap = float(np.mean(rev_vals >= pixel_threshold))
    else:
        rev_mean = rev_max = rev_overlap = 0.0

    # Match with reverse components
    rev_matches = match_components([poly], list(reverse_components))
    if 0 in rev_matches:
        match_info = rev_matches[0]
        rev_iou = float(match_info["iou"])
        rev_area = float(reverse_components[match_info["index"]].area)
    else:
        rev_iou = 0.0
        rev_area = 0.0

    forward_area = float(poly.area)
    prob_diff = rev_mean  # will be difference when combined: forward_mean - rev_mean
    area_ratio = forward_area / max(rev_area, 1e-4) if rev_area > 0 else 10.0

    return {
        "reverse_mean_prob": rev_mean,
        "reverse_max_prob": rev_max,
        "reverse_overlap_fraction": rev_overlap,
        "reverse_matching_iou": rev_iou,
        "prob_diff_forward_reverse": prob_diff,
        "area_ratio_forward_reverse": area_ratio,
        "reverse_asymmetry": 0.0,  # updated in orchestrator where forward_mean is available
    }


def compute_rgb_structural_features(
    y: np.ndarray,
    x: np.ndarray,
    ring_mask: np.ndarray,
    pre_rgb: np.ndarray,
    post_rgb: np.ndarray,
    global_metrics: dict[str, float],
    class_name: str,
) -> dict[str, float]:
    """Compute RGB, edge, texture and structural change inside candidate and ring."""
    # pre_rgb and post_rgb are (3, 256, 256) float in [0, 1]
    diff = np.abs(post_rgb - pre_rgb)  # (3, 256, 256)
    diff_mean_map = diff.mean(axis=0)  # (256, 256)

    # Luminance
    lum_pre = pre_rgb.mean(axis=0)
    lum_post = post_rgb.mean(axis=0)
    lum_diff_map = np.abs(lum_post - lum_pre)

    # Sobel edges on luminance
    dy_pre, dx_pre = np.gradient(lum_pre)
    dy_post, dx_post = np.gradient(lum_post)
    edge_pre = np.hypot(dx_pre, dy_pre)
    edge_post = np.hypot(dx_post, dy_post)
    edge_diff_map = np.abs(edge_post - edge_pre)

    # Local 3x3 variance
    mean_pre = ndi.uniform_filter(lum_pre, size=3)
    sq_pre = ndi.uniform_filter(lum_pre**2, size=3)
    var_pre = np.maximum(0.0, sq_pre - mean_pre**2)

    mean_post = ndi.uniform_filter(lum_post, size=3)
    sq_post = ndi.uniform_filter(lum_post**2, size=3)
    var_post = np.maximum(0.0, sq_post - mean_post**2)
    var_diff_map = np.abs(var_post - var_pre)

    if len(y) > 0:
        c_diff_mean = float(np.mean(diff_mean_map[y, x]))
        c_diff_r = float(np.mean(diff[0, y, x]))
        c_diff_g = float(np.mean(diff[1, y, x]))
        c_diff_b = float(np.mean(diff[2, y, x]))
        c_lum_diff = float(np.mean(lum_diff_map[y, x]))
        c_var_diff = float(np.mean(var_diff_map[y, x]))
        c_sobel_diff = float(np.mean(edge_diff_map[y, x]))
        c_edge_pre = float(np.mean(edge_pre[y, x] > 0.08))
        c_edge_post = float(np.mean(edge_post[y, x] > 0.08))
        c_edge_delta = c_edge_post - c_edge_pre

        # Local RGB histogram distance (32 bins per channel)
        h_pre, _ = np.histogram(lum_pre[y, x], bins=16, range=(0.0, 1.0), density=True)
        h_post, _ = np.histogram(lum_post[y, x], bins=16, range=(0.0, 1.0), density=True)
        local_hist_dist = float(0.5 * np.sum(np.abs(h_post - h_pre)) / max(len(h_pre), 1))
    else:
        c_diff_mean = c_diff_r = c_diff_g = c_diff_b = c_lum_diff = c_var_diff = 0.0
        c_sobel_diff = c_edge_pre = c_edge_post = c_edge_delta = local_hist_dist = 0.0

    # Ring statistics
    if ring_mask.any():
        r_diff_mean = float(np.mean(diff_mean_map[ring_mask]))
        r_sobel_diff = float(np.mean(edge_diff_map[ring_mask]))
    else:
        r_diff_mean = float(np.mean(diff_mean_map))
        r_sobel_diff = float(np.mean(edge_diff_map))

    rgb_ratio = c_diff_mean / max(r_diff_mean, 1e-4)
    rgb_diff_ring_diff = c_diff_mean - r_diff_mean
    sobel_ratio = c_sobel_diff / max(r_sobel_diff, 1e-4)

    # Contrast against global nuisance
    g_rgb = global_metrics["global_rgb_shift"]
    g_edge = global_metrics["global_edge_diff"]
    local_vs_global_rgb = c_diff_mean / max(g_rgb, 1e-4)
    local_vs_global_edge = c_sobel_diff / max(g_edge, 1e-4)

    features = {
        "rgb_diff_mean": c_diff_mean,
        "rgb_diff_r": c_diff_r,
        "rgb_diff_g": c_diff_g,
        "rgb_diff_b": c_diff_b,
        "luminance_diff": c_lum_diff,
        "local_variance_diff": c_var_diff,
        "sobel_mag_diff": c_sobel_diff,
        "edge_density_pre": c_edge_pre,
        "edge_density_post": c_edge_post,
        "edge_density_delta": c_edge_delta,
        "local_hist_dist": local_hist_dist,
        "ring_rgb_diff_mean": r_diff_mean,
        "rgb_ratio": rgb_ratio,
        "rgb_diff_ring_diff": rgb_diff_ring_diff,
        "sobel_ratio": sobel_ratio,
        "global_rgb_shift": g_rgb,
        "global_brightness_shift": global_metrics["global_brightness_shift"],
        "global_contrast_shift": global_metrics["global_contrast_shift"],
        "global_edge_diff": g_edge,
        "local_vs_global_rgb": local_vs_global_rgb,
        "local_vs_global_edge": local_vs_global_edge,
    }

    # Class-specific features
    if class_name == "new_building":
        features.update({
            "bld_post_edge_density": c_edge_post,
            "bld_pre_edge_density": c_edge_pre,
            "bld_edge_increase": max(0.0, c_edge_delta),
            "bld_structural_change": c_sobel_diff,
            "bld_rectangularity": 0.0,  # filled from geometry
            "bld_solidity": 0.0,  # filled from geometry
        })
    elif class_name == "tree_removal":
        # Excess green (ExG = 2*G - R - B)
        exg_pre_map = 2.0 * pre_rgb[1] - pre_rgb[0] - pre_rgb[2]
        exg_post_map = 2.0 * post_rgb[1] - post_rgb[0] - post_rgb[2]
        tot_pre = np.maximum(pre_rgb.sum(axis=0), 1e-4)
        tot_post = np.maximum(post_rgb.sum(axis=0), 1e-4)
        gr_pre_map = pre_rgb[1] / tot_pre
        gr_post_map = post_rgb[1] / tot_post

        if len(y) > 0:
            exg_pre_val = float(np.mean(exg_pre_map[y, x]))
            exg_post_val = float(np.mean(exg_post_map[y, x]))
            gr_pre_val = float(np.mean(gr_pre_map[y, x]))
            gr_post_val = float(np.mean(gr_post_map[y, x]))
        else:
            exg_pre_val = exg_post_val = gr_pre_val = gr_post_val = 0.0

        features.update({
            "tree_exg_pre": exg_pre_val,
            "tree_exg_post": exg_post_val,
            "tree_exg_drop": max(0.0, exg_pre_val - exg_post_val),
            "tree_green_ratio_pre": gr_pre_val,
            "tree_green_ratio_post": gr_post_val,
            "tree_green_ratio_drop": max(0.0, gr_pre_val - gr_post_val),
            "tree_texture_change": c_var_diff,
        })

    return features


def compute_global_metrics(pre_rgb: np.ndarray, post_rgb: np.ndarray) -> dict[str, float]:
    """Compute whole image pair appearance shift statistics."""
    rgb_diff = np.abs(post_rgb - pre_rgb).mean()
    lum_pre = pre_rgb.mean(axis=0)
    lum_post = post_rgb.mean(axis=0)
    b_shift = abs(float(lum_post.mean() - lum_pre.mean()))
    c_shift = abs(float(lum_post.std() - lum_pre.std()))

    dy_pre, dx_pre = np.gradient(lum_pre)
    dy_post, dx_post = np.gradient(lum_post)
    e_pre = np.hypot(dx_pre, dy_pre)
    e_post = np.hypot(dx_post, dy_post)
    e_diff = float(np.abs(e_post - e_pre).mean())

    return {
        "global_rgb_shift": float(rgb_diff),
        "global_brightness_shift": b_shift,
        "global_contrast_shift": c_shift,
        "global_edge_diff": e_diff,
    }


def compute_deep_cva_features(
    y: np.ndarray,
    x: np.ndarray,
    ring_mask: np.ndarray,
    cva_map: np.ndarray,
    cosine_dist_map: np.ndarray,
) -> dict[str, float]:
    """Compute deep encoder change vector features from Siamese features."""
    if len(y) > 0:
        c_vals = cva_map[y, x]
        cva_mean = float(np.mean(c_vals))
        cva_max = float(np.max(c_vals))
        cva_std = float(np.std(c_vals))
        cva_p90 = float(np.quantile(c_vals, 0.90))
        cos_dist = float(np.mean(cosine_dist_map[y, x]))
    else:
        cva_mean = cva_max = cva_std = cva_p90 = cos_dist = 0.0

    if ring_mask.any():
        r_vals = cva_map[ring_mask]
        ring_mean = float(np.mean(r_vals))
    else:
        ring_mean = float(np.mean(cva_map))

    cva_ratio = cva_mean / max(ring_mean, 1e-4)
    cva_diff = cva_mean - ring_mean

    return {
        "cva_mean": cva_mean,
        "cva_max": cva_max,
        "cva_std": cva_std,
        "cva_p90": cva_p90,
        "cva_ring_mean": ring_mean,
        "cva_ratio": cva_ratio,
        "cva_diff": cva_diff,
        "cva_cosine_dist": cos_dist,
    }


@torch.inference_mode()
def extract_deep_cva_maps(
    model: torch.nn.Module,
    pre_tensor: torch.Tensor,
    post_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 256x256 change magnitude and cosine distance maps from Siamese encoder.

    pre_tensor and post_tensor are shape (B, 3, 256, 256) normalized float tensors.
    """
    pre = pre_tensor.to(device)
    post = post_tensor.to(device)
    # ResNet18 encoder returns [feat0, feat1, feat2, feat3, feat4, feat5]
    # feat2 is stage 2: (B, 64, 64, 64)
    pre_feats = model.encoder(pre)
    post_feats = model.encoder(post)

    f_pre = pre_feats[2]
    f_post = post_feats[2]

    # L2 difference magnitude
    diff = torch.norm(f_post - f_pre, p=2, dim=1, keepdim=True)
    cva_map = F.interpolate(diff, size=(256, 256), mode="bilinear", align_corners=False)

    # Cosine distance: 1 - cos(f_pre, f_post)
    f_pre_n = F.normalize(f_pre, p=2, dim=1)
    f_post_n = F.normalize(post_feats[2], p=2, dim=1)
    cos_sim = (f_pre_n * f_post_n).sum(dim=1, keepdim=True)
    cos_dist = (1.0 - cos_sim).clamp(0.0, 2.0)
    cos_map = F.interpolate(cos_dist, size=(256, 256), mode="bilinear", align_corners=False)

    return cva_map.squeeze(1).cpu().numpy(), cos_map.squeeze(1).cpu().numpy()


def extract_candidate_evidence_record(
    poly: Polygon,
    component_index: int,
    y: np.ndarray,
    x: np.ndarray,
    all_polys: Sequence[Polygon],
    identity_prob: np.ndarray,
    identity_presence: float,
    verifier_score: float,
    stability_rec: Mapping[str, Any],
    reverse_rec: Mapping[str, Any] | None,
    cva_map: np.ndarray | None,
    cosine_map: np.ndarray | None,
    pre_rgb: np.ndarray | None,
    post_rgb: np.ndarray | None,
    global_metrics: dict[str, float] | None,
    class_name: str,
) -> dict[str, Any]:
    """Extract complete multi-group evidence dictionary for a single candidate component."""
    all_areas = [p.area for p in all_polys]
    geom = compute_geometry_features(poly, y, x, all_polys, component_index)
    conf = compute_confidence_features(y, x, identity_prob, identity_presence, verifier_score, all_areas)

    stab = {
        "persistence_count": float(stability_rec["persistence_count"]),
        "persistence_fraction": float(stability_rec["persistence_fraction"]),
        "mean_matching_iou": float(stability_rec["mean_matching_iou"]),
        "minimum_matching_iou": float(stability_rec["minimum_matching_iou"]),
        "maximum_matching_iou": float(max(stability_rec["matching_ious"][1:]) if len(stability_rec["matching_ious"]) > 1 else 0.0),
        "matching_area_mean": float(stability_rec["matching_component_area_mean"]),
        "area_std": float(stability_rec["area_std"]),
        "area_cv": float(stability_rec["area_coefficient_of_variation"]),
        "centroid_mean_drift": float(stability_rec["centroid_drift"]),
        "centroid_max_drift": float(stability_rec["maximum_centroid_drift"]),
        "tta_prob_mean": float(stability_rec["tta_probability_mean"]),
        "tta_prob_std": float(stability_rec["tta_probability_std"]),
    }

    if reverse_rec is not None:
        rev = dict(reverse_rec)
        forward_mean = conf["mean_probability"]
        rev_mean = rev["reverse_mean_prob"]
        rev["prob_diff_forward_reverse"] = forward_mean - rev_mean
        rev["reverse_asymmetry"] = abs(forward_mean - rev_mean) / max(forward_mean + rev_mean, 1e-4)
    else:
        rev = {k: 0.0 for k in REVERSE_FEATURES}

    ring_mask = get_surrounding_ring_mask(y, x, radius=5)

    if cva_map is not None and cosine_map is not None:
        cva = compute_deep_cva_features(y, x, ring_mask, cva_map, cosine_map)
    else:
        cva = {k: 0.0 for k in DEEP_CVA_FEATURES}

    if pre_rgb is not None and post_rgb is not None and global_metrics is not None:
        rgb_struct = compute_rgb_structural_features(
            y, x, ring_mask, pre_rgb, post_rgb, global_metrics, class_name
        )
        if class_name == "new_building":
            rgb_struct["bld_rectangularity"] = geom["rectangularity"]
            rgb_struct["bld_solidity"] = geom["solidity"]
    else:
        rgb_struct = {k: 0.0 for k in RGB_STRUCTURAL_FEATURES + GLOBAL_NUISANCE_FEATURES}
        if class_name == "new_building":
            rgb_struct.update({k: 0.0 for k in BUILDING_SPECIFIC_FEATURES})
        elif class_name == "tree_removal":
            rgb_struct.update({k: 0.0 for k in TREE_SPECIFIC_FEATURES})

    features = {}
    features.update(conf)
    features.update(geom)
    features.update(stab)
    features.update(rev)
    features.update(cva)
    features.update(rgb_struct)

    # Return record with polygon reference and complete feature map
    return {
        "polygon": stability_rec["polygon"],
        "component_index": component_index,
        "features": features,
    }
