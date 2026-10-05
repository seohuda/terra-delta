"""Finalize, package, and cleanroom-verify the TerraDelta v2.3.1 release."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import yaml

from terradelta.inference.v2_calibration import digest
from terradelta.inference.verifier import verify_frozen_checkpoint
from terradelta.submission import export_submission, make_submission_zip
from terradelta.utils.io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="/data/terradelta/v2.3.1/evaluation")
    parser.add_argument("--train-dir", default="/data/terradelta/v2.3.1/training")
    parser.add_argument("--checkpoint", default="/data/terradelta/releases/v2.2-main-01/model.pt")
    parser.add_argument("--base-config", default="/data/terradelta/releases/v2.2-main-01/config.yaml")
    parser.add_argument("--output", default="/data/terradelta/v2.3.1/release")
    parser.add_argument("--baseline-root", default="baseline/original/03_illegal structure submission")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    train_dir = Path(args.train_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify evaluation results and promotion gate
    comparison_file = eval_dir / "comparison.json"
    if not comparison_file.exists():
        raise FileNotFoundError(f"Missing evaluation comparison: {comparison_file}")
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))

    if not comparison.get("promotion_gate_passed"):
        raise ValueError(f"Promotion gate failed: {comparison.get('final_status')}")

    selected_ablation = comparison.get("selected_best_ablation", "D")
    gate_res = comparison["gate_results"][selected_ablation]
    print(f"Selected Best Ablation: {selected_ablation}")
    print(f"Gate Results: Real score={gate_res['real_score']:.6f}, FP={gate_res['no_change_fp']}, "
          f"B={gate_res['building_recall']}, T={gate_res['tree_recall']}, "
          f"Stress B_TP={gate_res['stress_building_tp']}, T_TP={gate_res['stress_tree_tp']}")

    # 2. Verify frozen checkpoint
    verify_frozen_checkpoint(args.checkpoint)
    ckpt_hash = digest(args.checkpoint)
    print(f"Checkpoint verified: {args.checkpoint} ({ckpt_hash})")

    # 3. Build self-contained v2.3.1 config with embedded classifier model
    base_config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    clf_file = train_dir / f"classifier_ablation_{selected_ablation}.json"
    clf_dict = json.loads(clf_file.read_text(encoding="utf-8"))

    v231_config = dict(base_config)
    # Remove v2.3 hard stability if present; v2.3.1 uses evidence classifier
    v231_config.pop("stability", None)
    v231_config["evidence_classifier"] = {
        "enabled": True,
        "ablation": selected_ablation,
        "model_data": clf_dict,
    }

    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(v231_config, sort_keys=False), encoding="utf-8")
    config_hash = digest(config_path)
    print(f"Config exported: {config_path} ({config_hash})")

    # 4. Export submission package
    package_dir = output_dir / "package"
    if package_dir.exists():
        shutil.rmtree(package_dir)

    print(f"Exporting submission package to {package_dir}...")
    export_submission(
        checkpoint=args.checkpoint,
        config_path=config_path,
        destination=package_dir,
        baseline_root=args.baseline_root,
    )

    # 5. Build submission zip
    zip_path = output_dir / "terradelta-v231-debug.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating submission zip at {zip_path}...")
    make_submission_zip(package_dir, zip_path)
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    zip_hash = digest(zip_path)
    print(f"ZIP created: {zip_path} ({zip_size_mb:.2f} MB, SHA256: {zip_hash})")

    if zip_size_mb > 100.0:
        raise ValueError(f"Package size {zip_size_mb:.2f} MB exceeds 100 MB limit")

    # 6. Cleanroom test: isolated execution in temporary directory
    print("\n--- Running Cleanroom Execution Test ---")
    cleanroom_dir = output_dir / "cleanroom"
    if cleanroom_dir.exists():
        shutil.rmtree(cleanroom_dir)
    cleanroom_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cleanroom_dir)

    # Verify no forbidden files in zip
    for p in cleanroom_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(cleanroom_dir))
            for forbidden in (".git", ".pyc", "__pycache__", "training", "test_", "val.csv"):
                if forbidden in rel:
                    raise ValueError(f"Forbidden file found in ZIP: {rel}")

    # Create mock inputs for cleanroom test
    mock_input_dir = cleanroom_dir / "mock_input"
    images_dir = mock_input_dir / "images"
    pair_ids = ["pair_01", "pair_02"]
    for pair_id in pair_ids:
        pdir = images_dir / pair_id
        pdir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)).save(pdir / "pre.png")
        Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)).save(pdir / "post.png")

    with (mock_input_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id"])
        for pid in pair_ids:
            writer.writerow([pid])

    # Run cleanroom inference pass 1
    pred1_path = cleanroom_dir / "prediction_1.csv"
    cmd = [
        sys.executable,
        "-c",
        f"""
import sys, os
from pathlib import Path
ROOT = Path('{cleanroom_dir}')
sys.path.insert(0, str(ROOT / 'assets/code'))
import torch
from terradelta.utils.io import load_config
from terradelta.inference.v231_predictor import predict_directory

CONFIG = load_config(ROOT / 'assets/config.yaml')
predict_directory(
    Path('{mock_input_dir}'),
    Path('{pred1_path}'),
    ROOT / 'assets/model/model.pt',
    CONFIG,
    device='cpu',
)
""",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cleanroom_dir / "assets/code")
    try:
        res1 = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Pass 1 STDOUT:", e.stdout)
        print("Pass 1 STDERR:", e.stderr)
        raise
    print("Cleanroom Pass 1 Output:", res1.stdout.strip())

    # Run cleanroom inference pass 2 (for byte-identical reproducibility)
    pred2_path = cleanroom_dir / "prediction_2.csv"
    cmd[2] = cmd[2].replace("prediction_1.csv", "prediction_2.csv")
    try:
        res2 = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Pass 2 STDOUT:", e.stdout)
        print("Pass 2 STDERR:", e.stderr)
        raise
    print("Cleanroom Pass 2 Output:", res2.stdout.strip())

    # Verify byte-identical CSVs
    hash1 = digest(pred1_path)
    hash2 = digest(pred2_path)
    if hash1 != hash2:
        raise ValueError(f"Reproducibility check failed: hash1={hash1} != hash2={hash2}")
    print(f"Cleanroom Verified: Predictions are byte-identical! (CSV SHA256: {hash1})")

    # 7. Write freeze receipt and production summary
    receipt = {
        "status": "V2.3.1 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL",
        "selected_ablation": selected_ablation,
        "checkpoint_sha256": ckpt_hash,
        "config_sha256": config_hash,
        "zip_sha256": zip_hash,
        "zip_size_bytes": zip_path.stat().st_size,
        "cleanroom_prediction_sha256": hash1,
        "gate_results": gate_res,
        "metrics_summary": {
            "legacy": {
                "score": gate_res["real_score"],
                "building_recall": gate_res["building_recall"],
                "tree_recall": gate_res["tree_recall"],
                "no_change_fp": gate_res["no_change_fp"],
            },
            "stress": {
                "score": gate_res["stress_score"],
                "building_tp": gate_res["stress_building_tp"],
                "tree_tp": gate_res["stress_tree_tp"],
            },
        },
    }
    receipt_path = output_dir / "v231-freeze-receipt.json"
    atomic_json(receipt_path, receipt)
    print(f"\nFreeze receipt written to {receipt_path}")
    print("\n=======================================================")
    print("V2.3.1 IMPROVED — READY FOR MAIN SUBMISSION APPROVAL")
    print("=======================================================")


if __name__ == "__main__":
    main()
