"""Future step-budget training. Importing this module never starts training."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Subset, default_collate

from .losses import build_loss
from .scheduler import lr_multiplier
from .callbacks import observe_metric


DEFAULT_MILESTONES = (25, 50, 75, 100, 150, 250, 500, 750, 1000)


def resolve_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Training supports CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def capture_rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    name, array, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(array, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact CUDA resume requires the original CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def isolated_rng(seed=None):
    """Keep validation/augmentation random draws out of the training RNG stream."""
    state = capture_rng_state()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            # CPU generator only: do not reset model CUDA dropout streams.
            torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)


class BatchStream:
    """Serializable epoch permutation and next-batch offset, without prefetch."""

    def __init__(self, size, batch_size, seed=0):
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset and batch_size must be positive")
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        self.epoch, self.offset = 0, 0
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(self.size, generator=self.generator).tolist()

    def remaining_batches(self):
        return math.ceil((self.size - self.offset) / self.batch_size)

    def next_batch(self):
        if self.offset == self.size:
            self.epoch += 1
            self.offset = 0
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
        start = self.offset
        self.offset = min(self.size, start + self.batch_size)
        return self.order[start:self.offset], self.epoch, start

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "seed": self.seed,
                "epoch": self.epoch, "offset": self.offset, "order": self.order.copy(),
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        for key in ("size", "batch_size", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Resume data-order {key} does not match")
        if sorted(state["order"]) != list(range(self.size)) or not 0 <= state["offset"] <= self.size or state["epoch"] < 0:
            raise ValueError("Invalid checkpoint permutation or epoch offset")
        if state["offset"] != self.size and state["offset"] % self.batch_size:
            raise ValueError("Checkpoint offset is not an optimizer-boundary batch offset")
        self.epoch, self.offset = state["epoch"], state["offset"]
        self.order = list(state["order"])
        self.generator.set_state(state["generator"].cpu())


def accumulation_batches(stream, accumulation_steps):
    """Take up to N batches without crossing an epoch boundary."""
    available = stream.remaining_batches() or math.ceil(stream.size / stream.batch_size)
    return [stream.next_batch() for _ in range(min(accumulation_steps, available))]


def validate_training_config(config):
    cfg = dict(config.get("training", {}))
    cfg.setdefault("max_optimizer_steps", 100)
    cfg.setdefault("batch_size", 4)
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("seed", 0)
    cfg.setdefault("encoder_lr", 1e-5)
    cfg.setdefault("head_lr", 1e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("amp", True)
    cfg.setdefault("deterministic", True)
    cfg.setdefault("num_workers", 0)
    cfg.setdefault("validate_every_steps", 25)
    cfg.setdefault("checkpoint_steps", list(DEFAULT_MILESTONES))
    cfg.setdefault("output_dir", "outputs/train_short")
    cfg.setdefault("scheduler", {"name": "constant", "warmup_steps": 0})
    cfg.setdefault("early_stopping", {"enabled": False})
    cfg.setdefault("max_skipped_updates", 100)
    for name in ("amp", "deterministic"):
        if not isinstance(cfg[name], bool):
            raise ValueError(f"training.{name} must be a boolean")
    if not isinstance(cfg["max_skipped_updates"], int) or cfg["max_skipped_updates"] < 0:
        raise ValueError("max_skipped_updates must be a nonnegative integer")
    for name in ("max_optimizer_steps", "batch_size", "gradient_accumulation_steps"):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if cfg["num_workers"] != 0:
        raise ValueError("Exact mid-epoch resume uses training.num_workers=0 (no hidden worker/prefetch RNG)")
    if not isinstance(cfg["seed"], int) or isinstance(cfg["seed"], bool) or cfg["seed"] < 0:
        raise ValueError("training.seed must be a nonnegative integer")
    for name in ("encoder_lr", "head_lr", "weight_decay"):
        if not math.isfinite(float(cfg[name])) or float(cfg[name]) < 0:
            raise ValueError(f"training.{name} must be finite and nonnegative")
    if not isinstance(cfg["validate_every_steps"], int) or cfg["validate_every_steps"] < 0:
        raise ValueError("validate_every_steps must be a nonnegative integer")
    if any(not isinstance(s, int) or isinstance(s, bool) or s < 1 for s in cfg["checkpoint_steps"]):
        raise ValueError("checkpoint_steps must contain positive optimizer-step integers")
    scheduler = cfg["scheduler"]
    if not isinstance(scheduler, dict) or scheduler.get("name", "constant") not in {"constant", "cosine", "linear"}:
        raise ValueError("scheduler.name must be constant, cosine or linear")
    warmup = scheduler.get("warmup_steps", 0)
    if not isinstance(warmup, int) or not 0 <= warmup < cfg["max_optimizer_steps"]:
        raise ValueError("warmup_steps must be an integer smaller than max_optimizer_steps")
    if not 0 <= float(scheduler.get("min_lr_ratio", 0.0)) <= 1:
        raise ValueError("min_lr_ratio must be in [0,1]")
    early = cfg["early_stopping"]
    if not isinstance(early, dict) or early.get("mode", "max") not in {"max", "min"}:
        raise ValueError("early_stopping must be a mapping with mode max or min")
    if early.get("enabled", False):
        if cfg["validate_every_steps"] == 0 or not early.get("metric"):
            raise ValueError("Early stopping requires validation and an explicit score metric")
        if not isinstance(early.get("patience", 5), int) or early.get("patience", 5) < 1:
            raise ValueError("early_stopping.patience must be a positive validation count")
        if not math.isfinite(float(early.get("min_delta", 0))) or float(early.get("min_delta", 0)) < 0:
            raise ValueError("early_stopping.min_delta must be finite and nonnegative")
    clip = cfg.get("max_grad_norm")
    if clip is not None and (not math.isfinite(float(clip)) or float(clip) <= 0):
        raise ValueError("max_grad_norm must be positive")
    # Validate losses without building an optimizer.
    build_loss(config)
    return cfg


def parameter_groups(model, cfg):
    encoder = getattr(model, "encoder", None)
    encoder_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
    enc = [p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids]
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    if not enc or not head:
        raise ValueError("Separate learning rates require trainable encoder and head parameters")
    return [{"params": enc, "lr": float(cfg["encoder_lr"]), "name": "encoder"},
            {"params": head, "lr": float(cfg["head_lr"]), "name": "head"}]


def _transform(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return getattr(dataset, "transform", None)


def _sample(dataset, index, seed):
    transform = _transform(dataset)
    # Albumentations maintains RNG objects separate from Python/NumPy globals.
    with isolated_rng(seed):
        if transform is not None:
            if hasattr(transform, "set_random_seed"):
                transform.set_random_seed(seed)
            elif hasattr(transform, "set_seed"):
                transform.set_seed(seed)
            elif hasattr(transform, "reseed"):
                transform.reseed(seed)
            elif not getattr(transform, "deterministic", False):
                raise ValueError("Transforms must expose set_random_seed/set_seed/reseed or deterministic=True for exact resume")
        return dataset[index]


def _batch(dataset, indices, epoch, start, seed):
    samples = [_sample(dataset, index, (seed + epoch * len(dataset) + start + position) % 2**32)
               for position, index in enumerate(indices)]
    return default_collate(samples)


def validate_batch(batch, ignore_index=-100):
    image, mask = batch["image"], batch["mask"]
    if image.ndim != 4 or image.shape[1] != 6 or not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("Dataset image must be finite floating-point B6HW")
    if mask.shape != (image.shape[0], *image.shape[2:]) or mask.dtype != torch.long:
        raise ValueError("Dataset mask must be torch.long BHW")
    if ((mask != ignore_index) & ((mask < 0) | (mask > 2))).any():
        raise ValueError("Dataset masks must contain 0/1/2 labels or configured ignore_index")
    return image, mask


def dry_run(model, train_dataset, val_dataset, config, device="auto"):
    """Forward/loss smoke and a plan. No optimizer or scheduler construction."""
    cfg = validate_training_config(config)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Train and validation partitions must both be nonempty")
    target = resolve_device(device)
    transform = _transform(train_dataset)
    if transform is not None and not any(hasattr(transform, attr) for attr in ("set_seed", "set_random_seed", "reseed")) and not getattr(transform, "deterministic", False):
        raise ValueError("Train transform must support reproducible per-sample seeds")
    # One sample from each partition. Construction in CLI is already offline.
    train_batch = _batch(train_dataset, [0], 0, 0, cfg["seed"])
    val_batch = _batch(val_dataset, [0], 0, 0, cfg["seed"])
    image, mask = validate_batch(train_batch)
    validate_batch(val_batch)
    modes = [(module, module.training) for module in model.modules()]
    model.to(target)
    try:
        model.eval()
        with isolated_rng(), torch.inference_mode():
            logits = model(image.to(target))
            loss = build_loss(config).to(target)(logits, mask.to(target))
        if not torch.isfinite(loss):
            raise ValueError("Smoke loss is not finite")
    finally:
        for module, mode in modes:
            module.training = mode
    return {"dry_run": True, "optimizer_constructed": False, "optimizer_steps_executed": 0,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "train_samples": len(train_dataset), "val_samples": len(val_dataset),
            "split_strategy": config.get("data", {}).get("split", {}).get("strategy", "region"),
            "device": str(target), "amp": bool(cfg["amp"]),
            "amp_dtype": "float16" if target.type == "cuda" else "bfloat16",
            "batch_size": cfg["batch_size"], "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "effective_batch_size": cfg["batch_size"] * cfg["gradient_accumulation_steps"],
            "max_optimizer_steps": cfg["max_optimizer_steps"],
            "checkpoint_steps": sorted(set(s for s in cfg["checkpoint_steps"] if s <= cfg["max_optimizer_steps"])),
            "encoder_lr": cfg["encoder_lr"], "head_lr": cfg["head_lr"],
            "scheduler": cfg["scheduler"], "output_dir": cfg["output_dir"],
            "smoke_logits_shape": list(logits.shape), "smoke_loss": float(loss),
            "resume": cfg.get("resume"), "seed": cfg["seed"]}


def _dataset_signature(dataset):
    if isinstance(dataset, Subset):
        return {"indices": list(dataset.indices), "base": _dataset_signature(dataset.dataset)}
    return {"type": type(dataset).__qualname__, "length": len(dataset),
            "samples": getattr(dataset, "samples", None)}


def resume_signature(config, cfg, dataset):
    keys = ("max_optimizer_steps", "batch_size", "gradient_accumulation_steps", "seed", "encoder_lr",
            "head_lr", "weight_decay", "amp", "deterministic", "scheduler", "max_grad_norm")
    signature = {"training": {k: cfg.get(k) for k in keys}, "loss": config.get("training", {}).get("loss", {}),
                 "model": config.get("model", {}), "data": config.get("data", {}),
                 "dataset": _dataset_signature(dataset), "manifests": {}}
    for key in ("manifest", "train_manifest", "val_manifest"):
        path = config.get("data", {}).get(key)
        if path and Path(path).is_file():
            signature["manifests"][key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class Trainer:
    """Only .fit() performs optimizer steps; checkpoints occur after full accumulation."""

    def __init__(self, model, train_dataset, val_dataset, config, device="auto", validation_fn=None):
        self.config = copy.deepcopy(config)
        self.cfg = validate_training_config(config)
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if not len(train_dataset) or not len(val_dataset):
            raise ValueError("Train/validation datasets must be nonempty")
        transform = _transform(train_dataset)
        if transform is not None and not any(hasattr(transform, attr) for attr in ("set_seed", "set_random_seed", "reseed")) and not getattr(transform, "deterministic", False):
            raise ValueError("Transform must support reproducible per-sample seeds")
        self.optimizer = torch.optim.AdamW(parameter_groups(model, self.cfg), weight_decay=float(self.cfg["weight_decay"]))
        sched = self.cfg["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: lr_multiplier(
            step, self.cfg["max_optimizer_steps"], sched.get("name", "constant"),
            sched.get("warmup_steps", 0), float(sched.get("min_lr_ratio", 0.0))))
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(self.cfg["amp"]) and self.device.type == "cuda")
        self.criterion = build_loss(config).to(self.device)
        self.stream = BatchStream(len(train_dataset), self.cfg["batch_size"], self.cfg["seed"])
        self.global_step, self.best_metric, self.bad_validations = 0, None, 0
        self.skipped_updates = 0
        self._accumulating = False
        self.signature = resume_signature(config, self.cfg, train_dataset)
        self.validation_fn = validation_fn
        self.output_dir = Path(self.cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.get("resume"):
            self.load_resume(self.cfg["resume"])

    def checkpoint_dict(self):
        if self._accumulating:
            raise RuntimeError("Checkpointing partial accumulated gradients is unsupported")
        return {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                "rng": capture_rng_state(), "data_order": self.stream.state_dict(),
                "resume_signature": self.signature, "training_format_version": 1,
                "steps": self.global_step, "seed": self.cfg["seed"], "device_type": self.device.type,
                "classes": ["background", "new_building", "tree_removal"], "in_channels": 6,
                "encoder": self.config.get("model", {}).get("encoder", "resnet18"),
                "best_metric": self.best_metric, "bad_validations": self.bad_validations,
                "skipped_updates": self.skipped_updates, "config": self.config,
                "runtime": {"torch": str(torch.__version__), "numpy": np.__version__}}

    def save_checkpoint(self, name):
        destination = self.output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.checkpoint_dict(), temporary)
        temporary.replace(destination)
        return destination

    def load_resume(self, path):
        # Our format contains only tensors and primitive containers, so safe loading works.
        state = torch.load(path, map_location="cpu", weights_only=True)
        required = {"optimizer", "scheduler", "scaler", "rng", "data_order", "resume_signature", "state_dict"}
        if not isinstance(state, dict) or not required <= state.keys():
            raise ValueError("Resume needs a TerraDelta training checkpoint; use init_checkpoint for baseline weights")
        if state.get("training_format_version") != 1 or state["resume_signature"] != self.signature:
            raise ValueError("Resume configuration/data differ from the saved training run")
        if state.get("device_type") != self.device.type:
            raise ValueError("Exact resume requires the original device type; use init_checkpoint to warm-start on another device")
        if state.get("runtime", {}).get("torch") != str(torch.__version__):
            raise ValueError("Exact resume requires the original PyTorch version")
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.stream.load_state_dict(state["data_order"])
        self.global_step = int(state["steps"])
        if not 0 <= self.global_step <= self.cfg["max_optimizer_steps"]:
            raise ValueError("Checkpoint step lies outside the configured budget")
        self.best_metric, self.bad_validations = state["best_metric"], int(state["bad_validations"])
        self.skipped_updates = int(state.get("skipped_updates", 0))
        restore_rng_state(state["rng"])

    def _log(self, record):
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")

    def _validate(self):
        if self.validation_fn is None:
            from .validation import validate_model
            self.validation_fn = validate_model
        mode = self.model.training
        try:
            self.model.eval()
            with isolated_rng(), torch.inference_mode():
                return self.validation_fn(self.model, self.val_dataset, self.config, device=str(self.device))
        finally:
            self.model.train(mode)

    def _observe(self, metrics):
        self.best_metric, self.bad_validations, improved, stop = observe_metric(
            metrics, self.cfg["early_stopping"], self.best_metric, self.bad_validations)
        return improved, stop

    def fit(self):
        """Future execution entry point; max budget counts successful optimizer updates."""
        self._log({"event": "start", "steps": self.global_step, "config": self.config,
                   "device": str(self.device), "runtime": {"torch": str(torch.__version__)},
                   "train_samples": len(self.train_dataset), "val_samples": len(self.val_dataset),
                   "resume_signature": self.signature})
        self.model.train()
        milestones = set(self.cfg["checkpoint_steps"])
        stop = False
        while self.global_step < self.cfg["max_optimizer_steps"] and not stop:
            # End-of-epoch partial accumulation is normalized by actual sample count.
            batches = accumulation_batches(self.stream, self.cfg["gradient_accumulation_steps"])
            count = len(batches)
            samples = sum(len(indices) for indices, _, _ in batches)
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            self._accumulating = True
            started = time.monotonic()
            for indices, epoch, start in batches:
                batch = _batch(self.train_dataset, indices, epoch, start, self.cfg["seed"])
                image, mask = validate_batch(batch, self.criterion.ignore_index)
                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
                                    enabled=bool(self.cfg["amp"])):
                    loss = self.criterion(self.model(image.to(self.device)), mask.to(self.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; no optimizer update performed for this group")
                fraction = len(indices) / samples
                self.scaler.scale(loss * fraction).backward()
                total_loss += float(loss.detach()) * fraction
            self.scaler.unscale_(self.optimizer)
            if self.cfg.get("max_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["max_grad_norm"]))
            if not self.scaler.is_enabled() and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise FloatingPointError("Nonfinite gradients; no optimizer update performed")
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulating = False
            if self.scaler.get_scale() < old_scale:
                self.skipped_updates += 1
                self._log({"event": "amp_skipped_update", "steps": self.global_step, "skipped_updates": self.skipped_updates})
                if self.skipped_updates > self.cfg.get("max_skipped_updates", 100):
                    self.save_checkpoint("last.pt")
                    raise FloatingPointError("Exceeded AMP skipped-update limit")
                continue
            self.global_step += 1
            used_lrs = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
            self.scheduler.step()
            self._log({"event": "train", "steps": self.global_step, "epoch": self.stream.epoch,
                       "epoch_offset": self.stream.offset, "loss": total_loss,
                       "samples": samples, "microbatches": count, "lr": used_lrs,
                       "seconds": time.monotonic() - started})
            interval = self.cfg["validate_every_steps"]
            if interval and (self.global_step % interval == 0 or self.global_step == self.cfg["max_optimizer_steps"]):
                metrics = self._validate()
                improved, stop = self._observe(metrics)
                self._log({"event": "validation", "steps": self.global_step, "metrics": metrics,
                           "best_metric": self.best_metric, "bad_validations": self.bad_validations})
                if improved:
                    self.save_checkpoint("best.pt")
            if self.global_step in milestones:
                self.save_checkpoint(f"step_{self.global_step:06d}.pt")
            self.save_checkpoint("last.pt")
        self.save_checkpoint("last.pt")
        result = {"steps": self.global_step, "early_stopped": stop,
                  "best_metric": self.best_metric, "checkpoint": str(self.output_dir / "last.pt")}
        self._log({"event": "complete", **result})
        return result
