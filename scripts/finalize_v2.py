"""Freeze selected v2 config, repeat actual inference, and export offline ZIP."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.inference.v2 import CLASSES, V2Predictor, output_row
from terradelta.inference.v2_calibration import MODEL, digest
from terradelta.metrics.evaluation import _polygon_union, evaluate_predictions
from terradelta.postprocess.polygons import mask_to_polygons_reference
from terradelta.submission import export_submission, make_submission_zip
from terradelta.utils.io import atomic_json, write_prediction_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--stress", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline-root", required=True)
    args = parser.parse_args()
    coarse = json.loads(Path(args.coarse).read_text())
    if coarse["status"] != "complete":
        raise ValueError("Coarse sweep is incomplete")
    selected = next(r for r in coarse["checkpoints"] if r["step"] == coarse["selected_step"])
    checkpoint = Path(selected["checkpoint"])
    if digest(checkpoint) != coarse["identity"]["checkpoint_sha256"][str(selected["step"])]:
        raise ValueError("Selected checkpoint changed")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "model": {**MODEL, "encoder_weights": None},
        "inference": {"batch_size": 8},
        "postprocess": {
            "mode": "independent",
            "min_area": 30,
            "min_pos_area": 20,
            "simplify_px": 0.5,
            "ndigits": 2,
            "classes": {
                n: {"presence_threshold": v["presence"], "pixel_threshold": v["pixel"]}
                for n, v in selected["selected_classes"].items()
            },
        },
    }
    config_path = root / "inference_v2_selected.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    predictor = V2Predictor(checkpoint, config, args.device)
    metrics = {}
    for name, manifest in [("legacy", args.legacy), ("stress", args.stress)]:
        data = IndependentChangeDataset(manifest)
        if digest(manifest) != coarse["identity"][name + "_manifest_sha256"]:
            raise ValueError("Validation manifest changed")
        predictions, truth = [], []
        for start in range(0, len(data), 8):
            samples = [data[i] for i in range(start, min(start + 8, len(data)))]
            pixels, presence = predictor.probabilities(torch.stack([s["image"] for s in samples]))
            predictions.extend(
                output_row(s["id"], p, q, config) for s, p, q in zip(samples, pixels, presence)
            )
            truth.extend(
                {
                    "id": s["id"],
                    **{
                        c: mask_to_polygons_reference(
                            s["target"][i].numpy() > 0, min_area=0, min_pos_area=0, simplify_px=0
                        )
                        for i, c in enumerate(CLASSES)
                    },
                }
                for s in samples
            )
        write_prediction_csv(root / (name + "-prediction.csv"), predictions)
        exact = evaluate_predictions(predictions, truth)
        nochange = [i for i, row in enumerate(truth) if all(not row[c] for c in CLASSES)]
        exact["no_change_count"] = len(nochange)
        exact["no_change_fp_any"] = sum(
            any(_polygon_union(predictions[i][c], "prediction").area >= 20 for c in CLASSES) for i in nochange
        )
        for c in CLASSES:
            actual = exact["classes"][c]
            expected = selected["selected_classes"][c][name]
            for key in ("score", "presence_macro_f1", "shape_score"):
                if not np.isclose(actual[key], expected[key], rtol=0, atol=1e-10):
                    raise ValueError(f"Final inference differs from cached sweep: {name}/{c}/{key}")
            if actual["presence_confusion"] != expected["presence_confusion"]:
                raise ValueError("Final presence confusion differs from cached sweep")
            actual["no_change_fp_count"] = expected["no_change_fp_count"]
        metrics[name] = exact
    selection = {
        "status": "validated",
        "step": selected["step"],
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint": str(checkpoint),
        "config_sha256": digest(config_path),
        "config": config,
        "selected_classes": selected["selected_classes"],
        "metrics": metrics,
        "identity": coarse["identity"],
        "selection_rule": coarse["selection_rule"],
        "final_inference_cache_parity": True,
        "training_steps_executed": 0,
        "main_submissions": 0,
        "legacy_is_real": True,
        "stress_is_synthetic": True,
    }
    atomic_json(root / "final-v2-selection.json", selection)
    atomic_json(Path(args.coarse).parent / "final-v2-selection.json", selection)
    package = export_submission(checkpoint, config_path, root / "package", args.baseline_root)
    archive = make_submission_zip(package, root / "terradelta-v2-debug.zip")
    repeat = make_submission_zip(package, root / "reproducibility-check.zip")
    if digest(archive) != digest(repeat):
        raise ValueError("ZIP reproducibility failed")
    print(
        json.dumps(
            {
                "status": "exported",
                "step": selected["step"],
                "metrics": metrics,
                "artifact_bytes": archive.stat().st_size,
                "artifact_sha256": digest(archive),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
