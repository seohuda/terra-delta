"""Pure optimizer-step LR schedule math; evaluated without training."""
import math


def lr_multiplier(step, max_steps, name="constant", warmup_steps=0, min_lr_ratio=0.0):
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if name == "constant":
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)))
    fraction = 0.5 * (1 + math.cos(math.pi * progress)) if name == "cosine" else 1 - progress
    return min_lr_ratio + (1 - min_lr_ratio) * fraction


