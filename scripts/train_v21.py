"""Two bounded consistency strengths from the frozen v2; no submission API."""

from collections import Counter
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import yaml

from terradelta.data.dataset_v2 import IndependentChangeDataset
from terradelta.data.season_v21 import SeasonTransform, photometric_view
from terradelta.inference.v2 import model_options
from terradelta.inference.v2_calibration import digest
from terradelta.inference.v2_evaluation import evaluate_model, promotion_gate
from terradelta.models.siamese_v2 import TerraDeltaSiameseV2, load_v2_checkpoint, save_v2_checkpoint
from terradelta.training.losses_v21 import V21Loss
from terradelta.training.sampling_v21 import draw_batch, pools
from terradelta.utils.io import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    config = yaml.safe_load(Path(a.config).read_text())
    settings, data = config["training"], config["data"]
    seed = int(config["seed"])
    checkpoint = settings["init_checkpoint"]
    expected = "ca59fe6698d8ee6bef354a7f40c1ef6d8901826e0572a5b80eb2b1d8b2ca4372"
    if digest(checkpoint) != expected:
        raise ValueError("Frozen v2 initialization changed")
    audit = json.loads(Path(data["audit"]).read_text())
    if audit.get("status") != "passed" or digest(data["train_manifest"]) != audit["train_sha256"]:
        raise ValueError("Training manifest lacks matching completed audit")
    for manifest in (data["legacy_val_manifest"], data["stress_val_manifest"]):
        if digest(manifest) != audit["input_sha256"][manifest]:
            raise ValueError("Validation manifest changed since data audit")
    strengths = settings["consistency_lambdas"]
    if strengths != [.05, .10] or settings["checkpoints"] != [25, 50, 100, 150]:
        raise ValueError("Only preregistered .05/.10 and <=150 fine-tune steps are authorized")
    device = torch.device(a.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Training must run on the approved GPU; no silent CPU fallback")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    infer = yaml.safe_load(Path(config["inference_config"]).read_text())
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    train = IndependentChangeDataset(data["train_manifest"], transform=SeasonTransform(**config["augmentation"]))
    groups = pools(train.rows)
    validation = {"legacy": data["legacy_val_manifest"], "stress": data["stress_val_manifest"]}
    base_model = TerraDeltaSiameseV2(**model_options(infer)).to(device)
    load_v2_checkpoint(checkpoint, base_model)
    baseline = {name: evaluate_model(base_model, m, infer, device) for name, m in validation.items()}
    atomic_json(output / "frozen-v2-reference.json", baseline)
    del base_model
    torch.cuda.empty_cache()
    runs = []
    for strength in strengths:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = random.Random(seed)
        run_root = output / f"lambda-{strength:.2f}"
        run_root.mkdir()
        model = TerraDeltaSiameseV2(**model_options(infer)).to(device)
        load_v2_checkpoint(checkpoint, model)
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        optimizer = torch.optim.AdamW([
            {"params": list(model.encoder.parameters()), "lr": settings["encoder_lr"]},
            {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": settings["new_lr"]},
        ], weight_decay=settings["weight_decay"])
        loss_fn = V21Loss(consistency_weight=strength, **settings["loss"])
        scaler = torch.amp.GradScaler("cuda")
        history, sampled = [], Counter()
        started = time.monotonic()
        for step in range(1, 151):
            model.train()
            # Keep v2's running statistics; two appearance forwards must not double BN updates.
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            indices = draw_batch(groups, settings["batch_size"], rng, settings["real_negative_ratio"])
            samples = [train.get(i, seed=seed + step * 10007 + j) for j, i in enumerate(indices)]
            sampled.update(s["category"] for s in samples)
            batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in (
                "image", "target", "valid_mask", "presence", "presence_valid",
            )}
            view = photometric_view(batch["image"], seed=seed + step * 65537)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                original, alternate = model(batch["image"]), model(view)
                loss, parts = loss_fn(original, alternate, batch["target"], batch["valid_mask"],
                                      batch["presence"], batch["presence_valid"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite training gradients")
            scaler.step(optimizer)
            scaler.update()
            if step in settings["checkpoints"]:
                path = run_root / f"step-{step:04d}.pt"
                save_v2_checkpoint(path, model, steps=250 + step, fine_tune_steps=step, seed=seed,
                                   experiment="v2.1-season-aware", consistency_lambda=strength,
                                   initialized_from_sha256=expected)
                metrics = {name: evaluate_model(model, m, infer, device, run_root / f"{name}-{step:04d}.csv")
                           for name, m in validation.items()}
                item = {"step": step, "lambda": strength, "checkpoint": str(path),
                        "checkpoint_sha256": digest(path), "metrics": metrics,
                        "gate": promotion_gate(metrics, baseline), "loss": float(loss.detach()),
                        "parts": {k: float(v) for k, v in parts.items()},
                        "sampled_categories": dict(sampled), "elapsed_seconds": time.monotonic() - started}
                history.append(item)
                atomic_json(run_root / "history.json", history)
                print(json.dumps(item, allow_nan=False), flush=True)
        runs.extend(history)
        del model, optimizer, batch, view, original, alternate, loss
        torch.cuda.empty_cache()
    eligible = [r for r in runs if r["gate"]["passed"]]
    ranked = eligible or runs
    chosen = max(ranked, key=lambda r: (-r["metrics"]["legacy"]["no_change_fp_any"],
                r["metrics"]["legacy"]["score"], r["metrics"]["stress"]["score"], -r["step"]))
    result = {"status": "improved" if eligible else "not_proven", "selected": chosen,
              "baseline": baseline, "runs": runs, "config_sha256": digest(a.config),
              "inference_config_sha256": digest(config["inference_config"]),
              "data_audit": audit, "main_submissions": 0, "debug_submissions": 0,
              "device": str(device), "gpu": torch.cuda.get_device_name(), "seed": seed}
    atomic_json(output / "comparison.json", result)
    print(json.dumps({"status": result["status"], "selected": chosen}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
