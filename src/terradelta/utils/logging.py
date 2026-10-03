"""Append-only metrics logs for future experiments."""
import json
import logging
from pathlib import Path


def get_logger(name="terradelta"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def log_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, allow_nan=False) + "\n")
