"""Bounded inference-only A-E comparison on existing, unchanged validation pairs.

Stage one evaluates 22 real pairs, five shared rules and at most one independent
class combination. Stage two evaluates at most two real-selected rules on the
existing stress set. One appearance arm is allowed only after geometric failure.
No training modules, optimizer, downloader or competition client are imported.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import torch
import yaml

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.inference.stability_v23 import (
    CLASSES, RULES, TRANSFORMS, apply_stability, component_features,
    radiometric_pair, tta_probabilities,
)
from terradelta.inference.v2 import V2Predictor, output_row
from terradelta.inference.v2_calibration import digest
from terradelta.inference.v2_evaluation import prediction_metrics
from terradelta.inference.verifier import verify_frozen_checkpoint
from terradelta.postprocess.polygons import mask_to_polygons_reference
from terradelta.utils.io import atomic_json, write_prediction_csv

PREFERENCE = ("A", "C", "D", "E", "B")


def stability_config(building, tree, appearance=False):
    return {"enabled": True, "appearance_consistency": appearance,
            "classes": dict(zip(CLASSES, (copy.deepcopy(RULES[building]), copy.deepcopy(RULES[tree]))))}


def local_gate(candidate, baseline):
    """Predeclared strong gate; no weakening after seeing the 22 real results."""
    if candidate["num_samples"] != 22 or candidate["no_change_count"] != 17:
        raise ValueError("Expected exactly 22 real validation pairs including 17 negatives")
    reasons = []
    for name, required in zip(CLASSES, (3, 2)):
        new, old = candidate["classes"][name], baseline["classes"][name]
        if new["gt_positive_count"] != required or new["presence_confusion"]["tp"] != required:
            reasons.append(f"{name}: recall must be {required}/{required}")
        if new["shape_score"] + 1e-10 < .95 * old["shape_score"]:
            reasons.append(f"{name}: shape score dropped more than 5 percent")
    if candidate["no_change_fp_any"] > 12:
        reasons.append("Real no-change FP must be <=12/17 (14/17 and 15/17 are rejected)")
    if candidate["score"] < .50:
        reasons.append("Real score must be >=0.50")
    return {"passed": not reasons, "reasons": reasons}


def stress_gate(candidate, baseline):
    """Serious collapse: >5% relative loss of TP count, shape or overall score."""
    if candidate["num_samples"] != baseline["num_samples"]:
        raise ValueError("Stress populations differ")
    reasons = []
    for name in CLASSES:
        new, old = candidate["classes"][name], baseline["classes"][name]
        if new["presence_confusion"]["tp"] < .95 * old["presence_confusion"]["tp"]:
            reasons.append(f"{name}: stress TP dropped more than 5 percent")
        if new["shape_score"] + 1e-10 < .95 * old["shape_score"]:
            reasons.append(f"{name}: stress shape dropped more than 5 percent")
    if candidate["score"] + 1e-10 < .95 * baseline["score"]:
        reasons.append("Stress overall score dropped more than 5 percent")
    return {"passed": not reasons, "reasons": reasons}


def _cache_identity(manifest, checkpoint, config, appearance, device):
    return {"manifest_sha256": digest(manifest), "checkpoint_sha256": digest(checkpoint),
            "config": config, "transforms": list(TRANSFORMS), "appearance": appearance,
            "device": device, "torch_version": torch.__version__,
            "inference_sha256": {name: digest(Path(__file__).parents[1] / "src/terradelta/inference" / name)
                                 for name in ("v2.py", "verifier.py", "stability_v23.py")},
            "matching": "same-class one-to-one IoU>=0.2 or distance<=4px and area ratio 1/4..4"}


@torch.inference_mode()
def extract(manifest, checkpoint, config, device, destination, appearance=False):
    identity = _cache_identity(manifest, checkpoint, config, appearance, device)
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached["identity"] != identity:
            raise ValueError("Existing feature diagnostic identity differs")
        if any(digest(path) != value for path, value in cached["input_hashes"].items()):
            raise ValueError("Cached validation inputs changed")
        return cached
    data = IndependentChangeDataset(manifest)
    predictor = V2Predictor(checkpoint, config, device)
    predictor.model.requires_grad_(False)
    frozen_config = {k: v for k, v in config.items() if k != "verifier"}
    batch_size = config.get("inference", {}).get("batch_size", 8)
    records, before_hashes = [], {}
    for row in data.rows:
        for key in ("pre", "post", *CLASSES, "valid_mask", "review_mask"):
            value = row.get(key)
            if value and value.lower() != "absent":
                before_hashes[value] = digest(value)
    started = time.monotonic()
    for start in range(0, len(data), batch_size):
        samples = [data[i] for i in range(start, min(start + batch_size, len(data)))]
        if any(not sample["valid_mask"].all() or not sample["presence_valid"].all() for sample in samples):
            raise ValueError("Validation requires fully reviewed complete labels")
        images = torch.stack([s["image"] for s in samples])
        pixels, heads = tta_probabilities(predictor, images)
        v22 = predictor.output_rows([s["id"] for s in samples], images, pixels[0], heads[0])
        normalized = predictor.probabilities(radiometric_pair(images)) if appearance else None
        for i, sample in enumerate(samples):
            record = {
                "id": sample["id"],
                "frozen": output_row(sample["id"], pixels[0, i], heads[0, i], frozen_config),
                "v22": v22[i],
                "truth": {"id": sample["id"], **{
                    name: mask_to_polygons_reference(sample["target"][c].numpy() > 0, 0, 0, 0)
                    for c, name in enumerate(CLASSES)}},
                "features": component_features(pixels[:, i], heads[:, i], config,
                    (normalized[0][i], normalized[1][i]) if normalized is not None else None),
            }
            # This equality binds all component decisions to the exact original
            # reference polygon order/vertices before evaluating any gate.
            for name in CLASSES:
                assert (json.loads(record["frozen"][name]) if record["frozen"][name] else []) == [
                    f["polygon"] for f in record["features"][name]]
            records.append(record)
        print(json.dumps({"event": "batch_complete", "samples": len(records), "total": len(data),
                          "seconds": time.monotonic() - started, "appearance": appearance}), flush=True)
    if any(digest(path) != value for path, value in before_hashes.items()):
        raise ValueError("Validation inputs changed during evaluation")
    verify_frozen_checkpoint(checkpoint)
    result = {"identity": identity, "records": records, "seconds": time.monotonic() - started,
              "source_files_verified_unchanged": len(before_hashes), "input_hashes": before_hashes,
              "training_steps": 0, "optimizer_backward_calls": 0}
    atomic_json(destination, result)
    return result


def evaluate(records, source, stability=None):
    truth = [r["truth"] for r in records]
    rows = [apply_stability(r[source], r["features"], stability) if stability else r[source] for r in records]
    return prediction_metrics(rows, truth), rows


def select_rules(trials, baseline):
    selected, plateaus = {}, {}
    for name in CLASSES:
        old = baseline["classes"][name]
        eligible = {key: row["combined"]["classes"][name] for key, row in trials.items()
                    if row["combined"]["classes"][name]["presence_confusion"]["tp"] == old["presence_confusion"]["tp"]
                    and row["combined"]["classes"][name]["shape_score"] >= .95 * old["shape_score"]}
        if not eligible:
            return None, {"reason": f"No geometric rule preserves {name} recall/shape"}
        minimum_fp = min(m["no_change_fp_count"] for m in eligible.values())
        best = max(m["score"] for m in eligible.values() if m["no_change_fp_count"] == minimum_fp)
        plateau = [key for key in PREFERENCE if key in eligible
                   and eligible[key]["no_change_fp_count"] == minimum_fp and eligible[key]["score"] >= best - .01]
        selected[name] = plateau[0]
        plateaus[name] = plateau
    return selected, plateaus


def rank_finalists(trials, baseline):
    """Stress only real-preserving rules; prefer the selected coarse plateau."""
    eligible = []
    for key, trial in trials.items():
        candidate = trial["combined"]
        if candidate["score"] < .50:
            continue
        if all(candidate["classes"][name]["presence_confusion"]["tp"]
               == baseline["classes"][name]["presence_confusion"]["tp"]
               and candidate["classes"][name]["shape_score"] >= .95 * baseline["classes"][name]["shape_score"]
               for name in CLASSES):
            eligible.append(key)
    ranked = sorted(eligible, key=lambda key: (not trials[key]["local_gate"]["passed"],
                    trials[key]["combined"]["no_change_fp_any"], -trials[key]["combined"]["score"]))
    if "class_selected" in ranked:
        choice, best = trials["class_selected"], trials[ranked[0]]
        if (choice["local_gate"]["passed"] == best["local_gate"]["passed"]
                and choice["combined"]["no_change_fp_any"] == best["combined"]["no_change_fp_any"]
                and choice["combined"]["score"] >= best["combined"]["score"] - .01):
            ranked.remove("class_selected")
            ranked.insert(0, "class_selected")
    return ranked


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    if "stability" in config or config.get("verifier") is None:
        raise ValueError("Input config must be unchanged v2.2 verifier config")
    if digest(args.config) != "0e0fd4bf695af2ef7c2b2aff9c5dc3b91babb1d4569f54b6ac416122ef9a1f77":
        raise ValueError("Expected exact v2.2 config bytes")
    verify_frozen_checkpoint(args.checkpoint)
    if digest(args.legacy) != "4e205c1a064b5bc315884ecc0d5ac92c17fbddfbb56918811cabf4589a71df77":
        raise ValueError("Real legacy manifest differs")
    if digest(args.stress) != "8bf7554944dce5cef1fff722490dbd72510a11a5a100ad2feadbe556bc3135bf":
        raise ValueError("Stress manifest differs")
    legacy = extract(args.legacy, args.checkpoint, config, args.device, root / "legacy-geometric.json")
    records = legacy["records"]
    frozen, _ = evaluate(records, "frozen")
    baseline, _ = evaluate(records, "v22")
    expected = json.loads(Path(args.v22_comparison).read_text())
    if frozen != expected["baseline"]["legacy"] or baseline != expected["candidate"]["legacy"]:
        raise ValueError("Identity evaluation does not reproduce frozen v2 and current best v2.2")
    trials = {}
    for key in RULES:
        stability = stability_config(key, key)
        only, _ = evaluate(records, "frozen", stability)
        combined, _ = evaluate(records, "v22", stability)
        trials[key] = {"rules": stability, "stability_only": only, "combined": combined,
                       "local_gate": local_gate(combined, baseline)}
    selected, plateaus = select_rules(trials, baseline)
    if selected:
        stability = stability_config(selected[CLASSES[0]], selected[CLASSES[1]])
        only, _ = evaluate(records, "frozen", stability)
        combined, _ = evaluate(records, "v22", stability)
        trials["class_selected"] = {"rules": stability, "stability_only": only, "combined": combined,
                                    "local_gate": local_gate(combined, baseline)}
    report = {"status": "geometric_evaluated", "current_best_main": .2045954238,
              "frozen": {"legacy": frozen}, "v22": {"legacy": baseline}, "trials": trials,
              "selected_classes": selected, "plateaus": plateaus, "transforms": list(TRANSFORMS),
              "inference_multiplier": 4, "legacy_seconds": legacy["seconds"],
              "main_submissions": 0, "debug_submissions": 0,
              "manifest_sha256": {"legacy": digest(args.legacy), "stress": digest(args.stress)},
              "selection_limitation": "Five coarse rules and one independent class combination reuse 22 real pairs; adaptive diagnostics, not independent generalization evidence."}
    atomic_json(root / "comparison.json", report)
    if args.legacy_only:
        print(json.dumps(report), flush=True)
        return
    # At most one additional appearance trial, only if geometry did not clear
    # the strong gate. It cannot rescue an already lost positive, so require
    # a geometric base preserving every real TP and >=95% shape first.
    if not any(t["local_gate"]["passed"] for t in trials.values()) and args.appearance:
        if selected:
            auxiliary = extract(args.legacy, args.checkpoint, config, args.device,
                                root / "legacy-appearance.json", appearance=True)
            stability = stability_config(selected[CLASSES[0]], selected[CLASSES[1]], True)
            only, _ = evaluate(auxiliary["records"], "frozen", stability)
            combined, _ = evaluate(auxiliary["records"], "v22", stability)
            trials["appearance_once"] = {"rules": stability, "stability_only": only, "combined": combined,
                                         "local_gate": local_gate(combined, baseline)}
            report["appearance_trial_count"] = 1
            report["appearance_seconds"] = auxiliary["seconds"]
        else:
            report["appearance_skipped"] = "Additional component veto cannot restore lost geometric positives/shape"
    ranked = rank_finalists(trials, baseline)
    if not ranked:
        report.update(status="not_proven", promotion_eligible=False,
                      stress_skipped="No rule preserves real recall/shape; no further inference justified")
        atomic_json(root / "comparison.json", report)
        print(json.dumps(report), flush=True)
        return
    # Stress only the best few, preferentially real-eligible rules. A duplicate
    # class-selected rule does not spend another stress evaluation slot.
    finalists, signatures = [], set()
    for key in ranked:
        signature = json.dumps(trials[key]["rules"], sort_keys=True)
        if signature not in signatures:
            finalists.append(key)
            signatures.add(signature)
        if len(finalists) == 2:
            break
    stress = extract(args.stress, args.checkpoint, config, args.device, root / "stress-geometric.json")
    stress_records = stress["records"]
    for source, target, original in (("frozen", "frozen", "baseline"), ("v22", "v22", "candidate")):
        score, _ = evaluate(stress_records, source)
        if score != expected[original]["stress"]:
            raise ValueError(f"Stress identity {source} differs from previous result")
        report[target]["stress"] = score
    passing = []
    for key in finalists:
        rule = trials[key]["rules"]
        if rule["appearance_consistency"]:
            stress_records = extract(args.stress, args.checkpoint, config, args.device,
                                     root / "stress-appearance.json", appearance=True)["records"]
        else:
            stress_records = stress["records"]
        only, _ = evaluate(stress_records, "frozen", rule)
        combined, predictions = evaluate(stress_records, "v22", rule)
        trials[key].update(stress_only=only, stress_combined=combined,
                           stress_gate=stress_gate(combined, report["v22"]["stress"]))
        write_prediction_csv(root / f"stress-{key}.csv", predictions)
        if trials[key]["local_gate"]["passed"] and trials[key]["stress_gate"]["passed"]:
            passing.append(key)
    best = passing[0] if passing else finalists[0]
    final_records = (json.loads((root / "legacy-appearance.json").read_text())["records"]
                     if trials[best]["rules"]["appearance_consistency"] else records)
    _, rows = evaluate(final_records, "v22", trials[best]["rules"])
    write_prediction_csv(root / "selected-legacy.csv", rows)
    deployment = copy.deepcopy(config)
    deployment["stability"] = trials[best]["rules"]
    (root / "selected-config.yaml").write_text(yaml.safe_dump(deployment, sort_keys=False))
    report.update(status="local_promotion_passed" if passing else "not_proven", selected=best,
                  stress_finalists=finalists, stress_seconds=stress["seconds"],
                  inference_multiplier=5 if trials[best]["rules"]["appearance_consistency"] else 4,
                  promotion_eligible=bool(passing), training_steps=0, optimizer_backward_calls=0)
    atomic_json(root / "comparison.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    def forbidden_training(*args, **kwargs):
        raise AssertionError("V2.3 evaluation forbids optimizer and backward")
    torch.optim.Optimizer.__init__ = forbidden_training
    torch.Tensor.backward = forbidden_training
    torch.autograd.backward = forbidden_training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--stress", required=True)
    parser.add_argument("--v22-comparison", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--appearance", action="store_true", help="Permit at most one appearance arm after weak geometry")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(parser.parse_args())
