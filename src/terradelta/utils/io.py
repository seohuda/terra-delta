"""Validated configuration and atomic local file I/O."""
from __future__ import annotations

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
