"""Frozen v2 feature extraction, verifier-only fit and exact polygon promotion gate.

No competition client or submission API. All bulk inputs/outputs remain on EC2.
"""

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.inference.v2 import CLASSES, V2Predictor, class_options, output_row
from terradelta.inference.v2_calibration import cache_probabilities, digest
from terradelta.inference.v2_evaluation import promotion_gate
from terradelta.inference.verifier import FEATURES, change_features, verifier_scores, verify_frozen_checkpoint
from terradelta.metrics.evaluation import _polygon_union, evaluate_predictions
from terradelta.postprocess.polygons import mask_to_polygons_reference
from terradelta.training.verifier import fit_verifier
from terradelta.utils.io import atomic_json, write_prediction_csv


def model_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def extract(data, manifest, predictor, config, output, cached=None):
    thresholds = [class_options(config, c)["pixel_threshold"] for c in CLASSES]
    identity = {
        "checkpoint_sha256": digest(args.checkpoint), "manifest_sha256": digest(manifest),
        "feature_extraction_sha256": hashlib.sha256((inspect.getsource(change_features) + repr(FEATURES)).encode()).hexdigest(),
        "config_sha256": digest(args.config),
    }
    if output.exists():
        z = np.load(output, allow_pickle=False)
        if json.loads(str(z["identity"])) != identity:
            raise ValueError("Feature cache identity differs")
        return {k: z[k] for k in z.files if k != "identity"}
    features, labels, valid, real, predictions, truth = [], [], [], [], [], []
    for start in range(0, len(data), 8):
        samples = [data[i] for i in range(start, min(start + 8, len(data)))]
        images = torch.stack([s["image"] for s in samples])
        if cached is None:
            pixels, presence = predictor.probabilities(images)
        else:
            pixels, presence = (v[start:start + len(samples)] for v in cached)
        features.extend(change_features(images.numpy(), pixels, presence, thresholds))
        labels.extend(s["presence"].numpy() for s in samples)
        valid.extend(s["presence_valid"].numpy() for s in samples)
        real.extend(data.rows[i].get("source") == "NAIP-real-reviewed" for i in range(start, start + len(samples)))
        if cached is not None:
            if any(not s["valid_mask"].all() for s in samples):
                raise ValueError("Full validation labels required")
            predictions.extend(output_row(s["id"], p, q, config) for s, p, q in zip(samples, pixels, presence))
            truth.extend({"id": s["id"], **{c: mask_to_polygons_reference(s["target"][j].numpy() > 0, 0, 0, 0)
                                         for j, c in enumerate(CLASSES)}} for s in samples)
        if start % 80 == 0:
            print(json.dumps({"event": "features", "manifest": str(manifest), "done": start + len(samples),
                              "total": len(data), "elapsed": time.monotonic() - begun}), flush=True)
    result = {"features": np.asarray(features), "labels": np.asarray(labels), "valid": np.asarray(valid),
              "real": np.asarray(real), "ids": np.asarray([r["id"] for r in data.rows])}
    if cached is not None:
        result.update(predictions=np.array(json.dumps(predictions)), truth=np.array(json.dumps(truth)))
    np.savez_compressed(output, **result, identity=json.dumps(identity, sort_keys=True))
    return result


def metrics(predictions, truth):
    result = evaluate_predictions(predictions, truth)
    negatives = [i for i, row in enumerate(truth) if all(not row[c] for c in CLASSES)]
    result["no_change_count"] = len(negatives)
    result["no_change_fp_any"] = sum(any(_polygon_union(predictions[i][c], "prediction").area >= 20
                                            for c in CLASSES) for i in negatives)
    for c in CLASSES:
        result["classes"][c]["no_change_fp_count"] = sum(
            _polygon_union(predictions[i][c], "prediction").area >= 20 for i in negatives)
    return result


def main():
    verify_frozen_checkpoint(args.checkpoint)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    audit = json.loads(Path(args.audit).read_text())
    if audit["status"] != "passed" or digest(args.train) != audit["train_sha256"]:
        raise ValueError("Expected existing audited training manifest")
    for manifest in (args.legacy, args.stress):
        if digest(manifest) != audit["input_sha256"][manifest]:
            raise ValueError("Held-out manifest changed")
    train = IndependentChangeDataset(args.train)
    if Counter(r["category"] for r in train.rows)["real_negative"] != 54:
        raise ValueError("Expected all 54 real negatives")
    # Reject unsafely added auxiliary rows: AIHub additions require a separate
    # provider/label/split audit, never a bare static land-cover presence label.
    if len(train) != audit["rows"]:
        raise ValueError("Training population differs from reviewed audit")
    predictor = V2Predictor(args.checkpoint, config, "cpu")
    predictor.model.requires_grad_(False).eval()
    original = model_digest(predictor.model)
    table = extract(train, args.train, predictor, config, root / "train-features.npz")
    verifier = fit_verifier(table["features"], table["labels"], table["valid"], table["real"],
                            [class_options(config, c)["pixel_threshold"] for c in CLASSES])
    baseline, candidate, held = {}, {}, {}
    for name, manifest in (("legacy", args.legacy), ("stress", args.stress)):
        data, pixels, presence = cache_probabilities(args.checkpoint, manifest,
                     Path(args.cache) / f"probabilities-step-0250-{name}.npz", "cpu")
        held[name] = extract(data, manifest, predictor, config, root / f"{name}-features.npz", (pixels, presence))
        del pixels, presence
        rows = json.loads(str(held[name]["predictions"]))
        truth = json.loads(str(held[name]["truth"]))
        baseline[name] = metrics(rows, truth)
        held[name]["scores"] = verifier_scores(held[name]["features"], verifier)
    # Fixed small grid, selected solely on local validation. Scores are adaptive
    # diagnostics, not an independent estimate. No DEBUG feedback is used.
    trials = []
    for c, name in enumerate(CLASSES):
        passing = []
        for threshold in sorted(set([0., .001, .005, .01, .015, .02, .025, *np.arange(.05, .951, .05)])):
            class_scores = []
            eligible = True
            for population in ("legacy", "stress"):
                table = held[population]
                rows = json.loads(str(table["predictions"]))
                for i, row in enumerate(rows):
                    if table["scores"][i, c] < threshold:
                        row[name] = ""
                result = metrics(rows, json.loads(str(table["truth"])))
                old, new = baseline[population]["classes"][name], result["classes"][name]
                if new["presence_confusion"]["tp"] < old["presence_confusion"]["tp"] or new["shape_score"] + 1e-10 < .95 * old["shape_score"] or new["score"] + 1e-10 < old["score"]:
                    eligible = False
                class_scores.append(new)
            row = {"class": name, "threshold": float(threshold), "eligible": eligible,
                   "legacy": class_scores[0], "stress": class_scores[1]}
            trials.append(row)
            if eligible:
                passing.append(row)
        if not passing:
            raise ValueError("Identity verifier should always pass per-class gate")
        selected = min(passing, key=lambda r: (r["legacy"]["no_change_fp_count"],
                             -(r["legacy"]["score"] + r["stress"]["score"]), r["threshold"]))
        verifier["threshold"][c] = selected["threshold"]
    for population in ("legacy", "stress"):
        table = held[population]
        rows = json.loads(str(table["predictions"]))
        for i, row in enumerate(rows):
            for c, name in enumerate(CLASSES):
                if table["scores"][i, c] < verifier["threshold"][c]:
                    row[name] = ""
        candidate[population] = metrics(rows, json.loads(str(table["truth"])))
        write_prediction_csv(root / f"{population}-prediction.csv", rows)
    if original != model_digest(predictor.model):
        raise ValueError("Frozen model state changed")
    verify_frozen_checkpoint(args.checkpoint)
    release = Path(args.checkpoint).parent
    hashes = json.loads((release / "SHA256SUMS.json").read_text())
    if any(digest(release / name) != value for name, value in hashes.items()):
        raise ValueError("Frozen release changed")
    config["verifier"] = verifier
    (root / "inference_v22.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    gate = promotion_gate(candidate, baseline)
    report = {"status": "local_gate_passed" if gate["passed"] else "not_proven", "gate": gate,
              "baseline": baseline, "candidate": candidate, "thresholds": verifier["threshold"],
              "training_rows": len(train), "real_negatives": 54, "aihub_rows": 0,
              "aihub_status": "awaiting_approved_access_and_sample_audit",
              "backbone_decoder_optimizer_steps": 0, "frozen_model_state_unchanged": True,
              "frozen_release_files_verified": len(hashes), "debug_submissions": 0, "main_submissions": 0,
              "elapsed_seconds": time.monotonic() - begun, "trials": trials,
              "selection_limitation": "Local threshold selection reuses 22 WA real and 470 synthetic validation pairs; not independent generalization evidence.",
              "identity": {"checkpoint_sha256": digest(args.checkpoint), "train_sha256": digest(args.train),
                           "config_sha256": digest(root / "inference_v22.yaml"),
                           "script_sha256": digest(__file__)}}
    atomic_json(root / "comparison.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--legacy", required=True)
    p.add_argument("--stress", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    begun = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    main()
