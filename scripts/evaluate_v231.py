"""Evaluation of object-level evidence classifier ablations A-E.

Evaluates bounded set of variants on legacy validation (22 pairs) and stress (470 pairs):
A. V2.3 current production baseline (hard stability: building 4/4, tree >=3/4)
B. V2.2 + classifier with confidence + geometry
C. B + TTA stability features
D. C + reverse-time features
E. D + deep CVA / local image evidence

Checks promotion gate:
- Real recall: building 3/3, tree 2/2
- Real no-change FP: <= 11/17 (prefer <= 10/17)
- Real score: >= 0.565 (prefer > 0.575)
- Kept polygon shapes remain original identity polygons
- Stress TP: building >= 111, tree >= 94 (prefer recovery toward V2.2: building >= 120, tree >= 100)
- Stress overall score: prefer >= 0.63
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.inference.evidence_classifier import EvidenceClassifier, apply_evidence_filtering
from terradelta.inference.evidence_features import (
    extract_candidate_evidence_record,
    extract_deep_cva_maps,
    get_component_pixels,
)
from terradelta.inference.stability_v23 import (
    apply_stability,
    component_features as stability_component_features,
    RULES,
    tta_probabilities,
)
from terradelta.inference.v2 import CLASSES, V2Predictor, class_options, output_row
from terradelta.inference.v2_calibration import digest
from terradelta.inference.v2_evaluation import prediction_metrics
from terradelta.inference.v231_predictor import (
    compute_global_metrics,
    compute_reverse_features,
    reverse_pair_tensor,
    tensor_to_rgb,
)
from terradelta.inference.verifier import verify_frozen_checkpoint
from terradelta.postprocess.polygons import mask_to_polygons_reference, reference_components
from terradelta.utils.io import atomic_json, write_prediction_csv


def evaluate_records(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [r["truth"] for r in records]
    return prediction_metrics(predictions, truth)


def evaluate_with_classifier(
    records: list[dict[str, Any]],
    classifier: EvidenceClassifier,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for r in records:
        filtered = apply_evidence_filtering(r["v22"], r["evidence_records"], classifier, config)
        rows.append(filtered)
    metrics = evaluate_records(records, rows)
    return metrics, rows


def evaluate_with_v23_stability(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Production V2.3 rules: building B (4/4), tree A (>=3/4)
    v23_rules = {
        "enabled": True,
        "appearance_consistency": False,
        "classes": {
            "new_building": copy.deepcopy(RULES["B"]),
            "tree_removal": copy.deepcopy(RULES["A"]),
        },
    }
    rows = []
    for r in records:
        filtered = apply_stability(r["v22"], r["stability_features"], v23_rules)
        rows.append(filtered)
    metrics = evaluate_records(records, rows)
    return metrics, rows


@torch.inference_mode()
def extract_validation_features(
    manifest: str | Path,
    checkpoint: str | Path,
    config: Mapping[str, Any],
    device: str,
    cache_path: Path,
) -> dict[str, Any]:
    """Extract and cache all evidence features for validation dataset."""
    identity = {
        "manifest_sha256": digest(manifest),
        "checkpoint_sha256": digest(checkpoint),
        "device": device,
        "torch_version": torch.__version__,
        "pipeline": "v2.3.1-object-evidence",
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("identity") == identity:
            print(f"Loading cached validation features from {cache_path}...")
            return cached

    print(f"Extracting validation features from {manifest} to {cache_path}...")
    data = IndependentChangeDataset(manifest)
    predictor = V2Predictor(checkpoint, config, device)
    frozen_config = {k: v for k, v in config.items() if k != "verifier"}
    batch_size = config.get("inference", {}).get("batch_size", 8)

    records = []
    started = time.monotonic()
    total = len(data)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        samples = [data[i] for i in range(start, end)]
        image_batch = torch.stack([s["image"] for s in samples])

        pixels_batch, presence_batch = predictor.probabilities(image_batch)
        v22_rows = predictor.output_rows(
            [s["id"] for s in samples], image_batch, pixels_batch, presence_batch
        )

        has_any_candidate = any(row.get(c) for row in v22_rows for c in CLASSES)
        if not has_any_candidate:
            for b_idx, sample in enumerate(samples):
                pair_id = sample["id"]
                records.append({
                    "id": pair_id,
                    "frozen": output_row(pair_id, pixels_batch[b_idx], presence_batch[b_idx], frozen_config),
                    "v22": v22_rows[b_idx],
                    "truth": {
                        "id": pair_id,
                        **{
                            name: mask_to_polygons_reference(sample["target"][c].numpy() > 0, 0, 0, 0)
                            for c, name in enumerate(CLASSES)
                        },
                    },
                    "stability_features": {c: [] for c in CLASSES},
                    "evidence_records": {c: [] for c in CLASSES},
                })
            if (start + batch_size) % 80 == 0 or end == total:
                print(f"Extracted {end}/{total} samples ({round(time.monotonic() - started, 1)}s)", flush=True)
            continue

        tta_pixels, tta_heads = tta_probabilities(predictor, image_batch, (pixels_batch, presence_batch))
        rev_images = reverse_pair_tensor(image_batch)
        rev_pixels, rev_heads = predictor.probabilities(rev_images)
        cva_maps, cos_maps = extract_deep_cva_maps(
            predictor.model, image_batch[:, :3], image_batch[:, 3:], predictor.device
        )
        pre_rgbs, post_rgbs = tensor_to_rgb(image_batch)

        for b_idx, sample in enumerate(samples):
            pair_id = sample["id"]
            global_metrics = compute_global_metrics(pre_rgbs[b_idx], post_rgbs[b_idx])
            stab_features = stability_component_features(
                tta_pixels[:, b_idx], tta_heads[:, b_idx], config
            )

            pair_evidence = {}
            for channel, class_name in enumerate(CLASSES):
                options = class_options(config, class_name)
                comp_list = stab_features[class_name]
                if not comp_list or not v22_rows[b_idx].get(class_name):
                    pair_evidence[class_name] = []
                    continue

                originals = [
                    p
                    for p in reference_components(pixels_batch[b_idx, channel] >= options["pixel_threshold"])
                    if p.area >= options["min_area"]
                ]
                if sum(p.area for p in originals) < options["min_pos_area"]:
                    pair_evidence[class_name] = []
                    continue

                rev_prob = rev_pixels[b_idx, channel]
                rev_pres = float(rev_heads[b_idx, channel])
                rev_parts = [
                    p
                    for p in reference_components(rev_prob >= options["pixel_threshold"])
                    if p.area >= options["min_area"]
                ]

                class_records = []
                for k, (part, stab_rec) in enumerate(zip(originals, comp_list)):
                    y, x = get_component_pixels(part)
                    rev_rec = compute_reverse_features(
                        y, x, part, rev_prob, rev_pres, rev_parts, options["pixel_threshold"]
                    )

                    rec = extract_candidate_evidence_record(
                        poly=part,
                        component_index=k,
                        y=y,
                        x=x,
                        all_polys=originals,
                        identity_prob=pixels_batch[b_idx, channel],
                        identity_presence=float(presence_batch[b_idx, channel]),
                        verifier_score=0.0,
                        stability_rec=stab_rec,
                        reverse_rec=rev_rec,
                        cva_map=cva_maps[b_idx],
                        cosine_map=cos_maps[b_idx],
                        pre_rgb=pre_rgbs[b_idx],
                        post_rgb=post_rgbs[b_idx],
                        global_metrics=global_metrics,
                        class_name=class_name,
                    )
                    class_records.append(rec)
                pair_evidence[class_name] = class_records

            record = {
                "id": pair_id,
                "frozen": output_row(pair_id, pixels_batch[b_idx], presence_batch[b_idx], frozen_config),
                "v22": v22_rows[b_idx],
                "truth": {
                    "id": pair_id,
                    **{
                        name: mask_to_polygons_reference(sample["target"][c].numpy() > 0, 0, 0, 0)
                        for c, name in enumerate(CLASSES)
                    },
                },
                "stability_features": stab_features,
                "evidence_records": pair_evidence,
            }
            records.append(record)

        if (start + batch_size) % 80 == 0 or end == total:
            print(f"Extracted {end}/{total} samples ({round(time.monotonic() - started, 1)}s)", flush=True)

    result = {
        "identity": identity,
        "records": records,
        "elapsed_sec": round(time.monotonic() - started, 1),
    }
    atomic_json(cache_path, result)
    return result


def analyze_feature_separation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Safety analysis: check distribution of reverse-time and CVA features on TP vs FP."""
    analysis = {}
    for name in CLASSES:
        tp_reverse_diff = []
        fp_reverse_diff = []
        tp_cva_ratio = []
        fp_cva_ratio = []

        for r in records:
            truth_has_class = bool(r["truth"].get(name))
            for rec in r["evidence_records"].get(name, []):
                feats = rec["features"]
                rev_diff = feats.get("prob_diff_forward_reverse", 0.0)
                cva_rat = feats.get("cva_ratio", 1.0)

                if truth_has_class:
                    tp_reverse_diff.append(rev_diff)
                    tp_cva_ratio.append(cva_rat)
                else:
                    fp_reverse_diff.append(rev_diff)
                    fp_cva_ratio.append(cva_rat)

        analysis[name] = {
            "n_tp_candidates": len(tp_reverse_diff),
            "n_fp_candidates": len(fp_reverse_diff),
            "tp_reverse_diff_mean": float(np.mean(tp_reverse_diff)) if tp_reverse_diff else 0.0,
            "fp_reverse_diff_mean": float(np.mean(fp_reverse_diff)) if fp_reverse_diff else 0.0,
            "tp_cva_ratio_mean": float(np.mean(tp_cva_ratio)) if tp_cva_ratio else 1.0,
            "fp_cva_ratio_mean": float(np.mean(fp_cva_ratio)) if fp_cva_ratio else 1.0,
            "reverse_diff_separates": (
                float(np.mean(tp_reverse_diff)) > float(np.mean(fp_reverse_diff)) + 0.05
                if tp_reverse_diff and fp_reverse_diff
                else False
            ),
            "cva_ratio_separates": (
                float(np.mean(tp_cva_ratio)) > float(np.mean(fp_cva_ratio)) + 0.10
                if tp_cva_ratio and fp_cva_ratio
                else False
            ),
        }
    return analysis


def check_promotion_gate(candidate_legacy, candidate_stress, baseline_v22, baseline_v23):
    """Check fixed promotion criteria from user prompt."""
    reasons = []

    # 1. Real recall
    bld_rec = candidate_legacy["classes"]["new_building"]["presence_confusion"]["tp"]
    tree_rec = candidate_legacy["classes"]["tree_removal"]["presence_confusion"]["tp"]
    if bld_rec != 3:
        reasons.append(f"Building legacy recall must be 3/3 (got {bld_rec}/3)")
    if tree_rec != 2:
        reasons.append(f"Tree legacy recall must be 2/2 (got {tree_rec}/2)")

    # 2. No-change FP
    fp_count = candidate_legacy["no_change_fp_any"]
    if fp_count > 11:
        reasons.append(f"Real no-change FP must be <= 11/17 (got {fp_count}/17)")

    # 3. Real score
    real_score = candidate_legacy["score"]
    if real_score < 0.565:
        reasons.append(f"Real score must be >= 0.565 (got {round(real_score, 6)})")

    # 4. Stress TP retention
    stress_bld_tp = candidate_stress["classes"]["new_building"]["presence_confusion"]["tp"]
    stress_tree_tp = candidate_stress["classes"]["tree_removal"]["presence_confusion"]["tp"]
    if stress_bld_tp < 111:
        reasons.append(f"Stress building TP must be >= 111 (got {stress_bld_tp})")
    if stress_tree_tp < 94:
        reasons.append(f"Stress tree TP must be >= 94 (got {stress_tree_tp})")

    passed = len(reasons) == 0
    return {
        "passed": passed,
        "reasons": reasons,
        "building_recall": f"{bld_rec}/3",
        "tree_recall": f"{tree_rec}/2",
        "no_change_fp": f"{fp_count}/17",
        "real_score": real_score,
        "stress_building_tp": stress_bld_tp,
        "stress_tree_tp": stress_tree_tp,
        "stress_score": candidate_stress["score"],
    }


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    verify_frozen_checkpoint(args.checkpoint)

    # 1. Extract/Load Legacy Validation
    legacy_cache = root / "legacy-evidence-cache.json"
    legacy_data = extract_validation_features(
        args.legacy, args.checkpoint, config, args.device, legacy_cache
    )
    legacy_records = legacy_data["records"]

    # Baseline evaluations
    frozen_legacy = evaluate_records(legacy_records, [r["frozen"] for r in legacy_records])
    v22_legacy = evaluate_records(legacy_records, [r["v22"] for r in legacy_records])
    v23_legacy, _ = evaluate_with_v23_stability(legacy_records)

    # Safety analysis of features on real legacy samples
    safety_analysis = analyze_feature_separation(legacy_records)

    # 2. Evaluate Ablations on Legacy
    classifiers = {}
    clf_dir = Path(args.classifier_dir)
    ablation_legacy_results = {}
    ablation_predictions = {}

    for key in ("B", "C", "D", "E"):
        model_path = clf_dir / f"classifier_ablation_{key}.json"
        if model_path.exists():
            clf = EvidenceClassifier.from_file(model_path)
            classifiers[key] = clf
            metrics, preds = evaluate_with_classifier(legacy_records, clf, config)
            ablation_legacy_results[key] = metrics
            ablation_predictions[key] = preds
            print(
                f"[Ablation {key}] Real score: {round(metrics['score'], 6)}, FP: {metrics['no_change_fp_any']}/17, "
                f"Recall: B={metrics['classes']['new_building']['presence_confusion']['tp']}/3, "
                f"T={metrics['classes']['tree_removal']['presence_confusion']['tp']}/2"
            )

    # 3. Extract/Load Stress Validation
    stress_cache = root / "stress-evidence-cache.json"
    stress_data = extract_validation_features(
        args.stress, args.checkpoint, config, args.device, stress_cache
    )
    stress_records = stress_data["records"]

    frozen_stress = evaluate_records(stress_records, [r["frozen"] for r in stress_records])
    v22_stress = evaluate_records(stress_records, [r["v22"] for r in stress_records])
    v23_stress, _ = evaluate_with_v23_stability(stress_records)

    ablation_stress_results = {}
    for key, clf in classifiers.items():
        metrics, preds = evaluate_with_classifier(stress_records, clf, config)
        ablation_stress_results[key] = metrics
        write_prediction_csv(root / f"stress-ablation-{key}.csv", preds)
        print(
            f"[Ablation {key} Stress] Score: {round(metrics['score'], 6)}, "
            f"B_TP: {metrics['classes']['new_building']['presence_confusion']['tp']}/150, "
            f"T_TP: {metrics['classes']['tree_removal']['presence_confusion']['tp']}/150, "
            f"FP: {metrics['no_change_fp_any']}/200"
        )

    # Check promotion gate for each ablation
    gate_results = {}
    for key in classifiers:
        gate_results[key] = check_promotion_gate(
            ablation_legacy_results[key],
            ablation_stress_results[key],
            v22_legacy,
            v23_legacy,
        )

    # Select best candidate
    passing_keys = [k for k, g in gate_results.items() if g["passed"]]
    if passing_keys:
        # Prefer higher stress TP and lower FP
        best_candidate = max(
            passing_keys,
            key=lambda k: (
                -ablation_legacy_results[k]["no_change_fp_any"],
                ablation_stress_results[k]["classes"]["new_building"]["presence_confusion"]["tp"],
                ablation_legacy_results[k]["score"],
            ),
        )
        final_status = "V2.3.1 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL"
    else:
        # Fall back to best key
        best_candidate = max(
            classifiers.keys(),
            key=lambda k: (
                -ablation_legacy_results[k]["no_change_fp_any"],
                ablation_legacy_results[k]["score"],
            ),
        )
        final_status = "V2.3.1 NOT PROVEN — KEEP V2.3"

    report = {
        "final_status": final_status,
        "selected_best_ablation": best_candidate,
        "promotion_gate_passed": bool(passing_keys),
        "gate_results": gate_results,
        "safety_analysis": safety_analysis,
        "baselines": {
            "frozen_v2": {"legacy": frozen_legacy, "stress": frozen_stress},
            "v22": {"legacy": v22_legacy, "stress": v22_stress},
            "v23": {"legacy": v23_legacy, "stress": v23_stress},
        },
        "ablations": {
            k: {"legacy": ablation_legacy_results[k], "stress": ablation_stress_results[k]}
            for k in classifiers
        },
    }

    atomic_json(root / "comparison.json", report)
    print(f"\nFinal Status: {final_status}")
    print(f"Selected: Ablation {best_candidate}")
    print(f"Full comparison written to {root / 'comparison.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--stress", required=True)
    parser.add_argument("--classifier-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    main(parser.parse_args())
