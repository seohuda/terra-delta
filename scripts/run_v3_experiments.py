#!/usr/bin/env python3
"""Run and orchestrate TerraDelta V3 Satlas Experiments (S1, S2, Evaluation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n>>> Running: {desc}...")
    print(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f">>> Completed {desc} in {time.time()-t0:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Orchestrate TerraDelta V3 S1, S2, and Evaluation")
    parser.add_argument("--skip-train-s1", action="store_true")
    parser.add_argument("--skip-train-s2", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/v3_satlas")
    parser.add_argument("--manifests-dir", type=Path, default=REPO_ROOT / "data/v3_manifests")
    parser.add_argument("--satlas-weights", type=Path, default=REPO_ROOT / "models/satlas/aerial_swinb_si.pth")
    parser.add_argument("--legacy-manifest", type=Path,
                        default=REPO_ROOT / "data/processed/micro-pilot-v2/val.csv")
    parser.add_argument("--stress-manifest", type=Path,
                        default=REPO_ROOT / "data/v2/manifests/stress-val-v2.csv")
    args = parser.parse_args()

    # 1. Train S1 if not skipped
    if not args.skip_train_s1:
        run_cmd([
            args.python, str(args.base_dir / "scripts/train_satlas_v3.py"),
            "--recipe", "S1",
            "--manifests-dir", str(args.manifests_dir),
            "--satlas-weights", str(args.satlas_weights),
            "--output-dir", str(args.output_dir),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--save-interval", "200",
        ], "S1 Training (+ AIHub Positives)")

    # 2. Train S2 if not skipped
    if not args.skip_train_s2:
        run_cmd([
            args.python, str(args.base_dir / "scripts/train_satlas_v3.py"),
            "--recipe", "S2",
            "--manifests-dir", str(args.manifests_dir),
            "--satlas-weights", str(args.satlas_weights),
            "--output-dir", str(args.output_dir),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--save-interval", "200",
        ], "S2 Training (+ Diverse AIHub Negatives)")

    # 3. Evaluate all recipes and steps
    recipes = ["S0", "S1", "S2"]
    eval_results = {}
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for r in recipes:
        recipe_dir = args.output_dir / r
        if not recipe_dir.exists():
            continue

        ckpts = sorted(recipe_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if (recipe_dir / "model.pt").exists() and (recipe_dir / "model.pt") not in ckpts:
            ckpts.append(recipe_dir / "model.pt")

        for ckpt in ckpts:
            step_name = ckpt.stem
            report_out = reports_dir / f"eval_{r}_{step_name}.json"
            print(f"Evaluating {r} {step_name}...")
            try:
                cmd = [
                    args.python, str(args.base_dir / "scripts/evaluate_satlas_v3.py"),
                    "--checkpoint", str(ckpt),
                    "--output", str(report_out),
                    "--legacy-manifest", str(args.legacy_manifest),
                    "--stress-manifest", str(args.stress_manifest),
                ]
                subprocess.run(cmd, check=True)
                with open(report_out) as f:
                    eval_results[f"{r}_{step_name}"] = json.load(f)
            except Exception as e:
                print(f"Error evaluating {ckpt}: {e}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Experiment':<25} | {'Real Score':<10} | {'Bldg Rec':<8} | {'Tree Rec':<8} | {'Real FP':<8} | {'Stress Score':<12} | {'Stress B_TP':<11} | {'Gates'}")
    print("-" * 90)
    for k, v in eval_results.items():
        leg = v["legacy"]
        str_res = v["stress"]
        gates = "PASS" if v["gates"]["passed_all"] else "FAIL"
        print(
            f"{k:<25} | {leg['score']:<10.4f} | {leg['bldg_tp']}/{leg['bldg_positives']:<5} | {leg['tree_tp']}/{leg['tree_positives']:<5} | {leg['no_change_fp']}/{leg['no_change_count']:<5} | {str_res['score']:<12.4f} | {str_res['bldg_tp']:<11} | {gates}"
        )
    print("=" * 90 + "\n")

    summary_file = reports_dir / "all_experiments_summary.json"
    with open(summary_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved complete experiment summary to {summary_file}")


if __name__ == "__main__":
    main()
