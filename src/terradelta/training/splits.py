"""Training partitions use the data pipeline's single geographic/license policy."""
from pathlib import Path

from torch.utils.data import Subset

from terradelta.data.split import assert_no_spatial_leak, split_manifest
from terradelta.data.pairing import training_eligibility
from terradelta.utils.io import read_csv, safe_id


def split_indices(rows, strategy="region", val_fraction=0.2, seed=0, *, buffer_m=0.0, temporal_year=None):
    """Adapter preserving indices while all split invariants live in data.split."""
    rows = [{**row, "_split_index": i} for i, row in enumerate(rows)]
    parts = split_manifest(rows, strategy=strategy, val_fraction=val_fraction, seed=seed,
                           buffer_m=buffer_m, temporal_year=temporal_year)
    return [row["_split_index"] for row in parts["train"]], [row["_split_index"] for row in parts["val"]]


def manifest_rows(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Grouped splitting requires a CSV manifest with region/state/year metadata")
    rows = read_csv(path)
    ids = [safe_id(row.get("id", "")) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest ids must be nonempty and unique")
    for row, identifier in zip(rows, ids):
        row["id"] = identifier
        for key in ("pre", "post", "new_building", "tree_removal"):
            value = str(row.get(key, "")).strip()
            if value and value.lower() != "absent":
                row[key] = str((path.parent / value).resolve())
    return rows


def assert_training_licenses(rows, config):
    """Reviewed provenance is checked before reading images; no legal inference."""
    if not config.get("data", {}).get("require_license_approval", True):
        return
    rejected = [(row.get("id", "?"), training_eligibility(row)[1])
                for row in rows if not training_eligibility(row)[0]]
    if rejected:
        raise ValueError(f"Training manifest contains unapproved provenance/labels: {rejected[:5]}; prepare a reviewed manifest first")


def prepare_datasets(config, dataset_factory=None, train_transform=None):
    if dataset_factory is None:
        from terradelta.data.dataset import ChangeDataset
        dataset_factory = ChangeDataset
    section = config.get("data", {})
    split = section.get("split", {})
    strategy = split.get("strategy", "region")
    if strategy not in {"region", "state", "temporal", "random"}:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    train_path, val_path = section.get("train_manifest"), section.get("val_manifest")
    if bool(train_path) != bool(val_path):
        raise ValueError("Set both data.train_manifest and data.val_manifest, or neither")
    if train_path:
        train_rows, val_rows = manifest_rows(train_path), manifest_rows(val_path)
        assert_training_licenses(train_rows, config)
        assert_training_licenses(val_rows, config)
        if not train_rows or not val_rows:
            raise ValueError("Both explicit manifests must be nonempty")
        if {r["id"] for r in train_rows} & {r["id"] for r in val_rows}:
            raise ValueError("Train and validation ids overlap")
        # State grouping also goes through the canonical spatial graph.
        from terradelta.data.split import spatial_groups
        groups = spatial_groups(train_rows + val_rows, strategy=strategy, buffer_m=split.get("buffer_m", 0))
        if any(any(i < len(train_rows) for i in g) and any(i >= len(train_rows) for i in g) for g in groups):
            raise ValueError(f"Explicit manifests leak {strategy} groups or adjacent tiles between partitions")
        assert_no_spatial_leak(train_rows, val_rows, buffer_m=split.get("buffer_m", 0))
        if strategy == "temporal":
            expected = split_manifest(train_rows + val_rows, strategy="temporal",
                temporal_year=split.get("temporal_year"), buffer_m=split.get("buffer_m", 0))
            if {r["id"] for r in expected["train"]} != {r["id"] for r in train_rows}:
                raise ValueError("Explicit temporal manifests do not follow the configured year cutoff")
        return (dataset_factory(train_path, transform=train_transform, require_masks=True),
                dataset_factory(val_path, transform=None, require_masks=True))
    path = section.get("manifest")
    if not path:
        raise ValueError("Configure data.manifest or train_manifest and val_manifest")
    rows = manifest_rows(path)
    assert_training_licenses(rows, config)
    train, val = split_indices(rows, strategy, split.get("val_fraction", 0.2),
                               split.get("seed", config.get("training", {}).get("seed", 0)),
                               buffer_m=split.get("buffer_m", 0), temporal_year=split.get("temporal_year"))
    train_data = dataset_factory(path, transform=train_transform, require_masks=True)
    val_data = dataset_factory(path, transform=None, require_masks=True)
    if len(train_data) != len(rows) or len(val_data) != len(rows):
        raise ValueError("Dataset ordering/length must match manifest rows")
    return Subset(train_data, train), Subset(val_data, val)
