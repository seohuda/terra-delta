"""Repeat eligible v2.2 inference, export frozen weights, and prove offline execution."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from terradelta.inference.v2 import V2Predictor
from terradelta.inference.v2_calibration import digest
from terradelta.inference.v2_evaluation import evaluate_model, promotion_gate
from terradelta.inference.verifier import verify_frozen_checkpoint
from terradelta.submission import export_submission, make_submission_zip
from terradelta.utils.io import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--legacy", required=True)
    p.add_argument("--stress", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline-root", required=True)
    a = p.parse_args()
    fit = Path(a.fit_root)
    report = json.loads((fit / "comparison.json").read_text())
    if report["status"] != "local_gate_passed" or not report["gate"]["passed"]:
        raise ValueError("Local promotion gate must pass before packaging")
    if not report["frozen_model_state_unchanged"] or report["backbone_decoder_optimizer_steps"] != 0:
        raise ValueError("Backbone/decoder must remain frozen")
    verify_frozen_checkpoint(a.checkpoint)
    config_path = fit / "inference_v22.yaml"
    if digest(config_path) != report["identity"]["config_sha256"]:
        raise ValueError("Selected verifier configuration changed")
    config = yaml.safe_load(config_path.read_text())
    frozen_config = yaml.safe_load((Path(a.checkpoint).parent / "inference_v2_selected.yaml").read_text())
    if {k: v for k, v in config.items() if k != "verifier"} != frozen_config:
        raise ValueError("Frozen pixel/presence/geometry settings changed")
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    predictor = V2Predictor(a.checkpoint, config, "cpu")
    predictor.model.requires_grad_(False)
    actual = {}
    for name, manifest in (("legacy", a.legacy), ("stress", a.stress)):
        actual[name] = evaluate_model(predictor.model, manifest, config, "cpu", output / f"{name}-prediction.csv")
        if actual[name] != report["candidate"][name]:
            raise ValueError(f"Actual inference differs from feature-table evaluation: {name}")
        print(json.dumps({"event": "actual_inference_verified", "population": name,
                          "score": actual[name]["score"], "no_change_fp": actual[name]["no_change_fp_any"]}), flush=True)
    if not promotion_gate(actual, report["baseline"])["passed"]:
        raise ValueError("Actual inference failed promotion")
    for name, expected in (("new_building", 3), ("tree_removal", 2)):
        c = actual["legacy"]["classes"][name]
        if c["gt_positive_count"] != expected or c["presence_confusion"]["tp"] != expected:
            raise ValueError("Real building3/3 and tree2/2 recall are mandatory")
    package = export_submission(a.checkpoint, config_path, output / "package", a.baseline_root)
    archive = make_submission_zip(package, output / "terradelta-v2-debug.zip")
    repeat = make_submission_zip(package, output / "reproducibility-check.zip")
    if digest(archive) != digest(repeat):
        raise ValueError("Submission ZIP is not reproducible")
    selection = {"status": "validated", "step": 250, "checkpoint_sha256": digest(a.checkpoint),
                 "config_sha256": digest(package / "assets/config.yaml"), "metrics": actual,
                 "baseline": report["baseline"], "gate": report["gate"],
                 "backbone_decoder_optimizer_steps": 0, "verifier_only": True}
    atomic_json(output / "final-v2-selection.json", selection)
    subprocess.run([sys.executable, str(Path(__file__).with_name("verify_v2_zip.py")),
                    "--package-root", str(output), "--legacy", a.legacy, "--stress", a.stress], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(version="v2.2", verifier_only=True, local_gate=report["gate"],
                    backbone_decoder_optimizer_steps=0, aihub_rows=report["aihub_rows"],
                    aihub_status=report["aihub_status"], main_submissions=0)
    atomic_json(output / "manifest.json", manifest)
    verify_frozen_checkpoint(a.checkpoint)
    print(json.dumps({"status": "debug_ready", "artifact": str(archive), "sha256": digest(archive),
                      "gate": report["gate"], "cleanroom": manifest["cleanroom"]}), flush=True)


if __name__ == "__main__":
    main()
