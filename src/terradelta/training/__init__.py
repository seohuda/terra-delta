"""Training building blocks. No training runs on import."""
from .losses import SegmentationLoss, build_loss
from .splits import prepare_datasets, split_indices
from .trainer import BatchStream, Trainer, dry_run

__all__ = ["BatchStream", "Trainer", "SegmentationLoss", "build_loss", "dry_run", "prepare_datasets", "split_indices"]
