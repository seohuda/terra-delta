"""Validated configuration and atomic local file I/O."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import tempfile

import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_prediction_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "new_building", "tree_removal"]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction ids")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_id(value):
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid sample id: {value!r}")
    return value


@contextmanager
def training_run_lock(directory, *, resume=False):
    """Protect a future local run from concurrent writers and accidental overwrite.

    Linux flock releases on process exit; the lock inode is kept to prevent races.
    No callers from dry-run or inference create a training output directory.
    """
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Training output directory is in use: {directory}") from error
        try:
            if not resume and ((directory / "metrics.jsonl").exists() or any(directory.glob("*.pt"))):
                raise FileExistsError(f"Existing run in {directory}; use --resume or choose a new output directory")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
