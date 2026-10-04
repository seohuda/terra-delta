"""Audit existing reviewed data and prepare a new v2.1 manifest without editing inputs."""

import argparse
from collections import Counter
import csv
from functools import cache
import hashlib
import json
from pathlib import Path

from terradelta.data.dataset_v2 import IndependentChangeDataset, read_v2_manifest


@cache
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(train_manifest, review_manifest, validation_manifests, output):
    train = IndependentChangeDataset(train_manifest)
    reviews = {r["id"]: r for r in read_v2_manifest(review_manifest)}
    held_out = [r for manifest in validation_manifests for r in read_v2_manifest(manifest)]
    held_ids = {r.get("source_id") or r["id"] for r in held_out}
    held_regions = {r["region_id"] for r in held_out}
    held_paths = {r[k] for r in held_out for k in ("pre", "post")}
    held_hashes = {digest(p) for p in held_paths}
    real_counts = Counter()
    categories = Counter()
    rows = []
    for index, row in enumerate(train.rows):
        if row.get("region_id") in held_regions:
            raise ValueError(f"Held-out region in training: {row['id']}")
        source_id = row.get("source_id") or row["id"]
        if source_id in held_ids:
            raise ValueError(f"Held-out source in training: {source_id}")
        if any(row[k] in held_paths or digest(row[k]) in held_hashes for k in ("pre", "post")):
            raise ValueError(f"Validation image leakage: {row['id']}")
        if row.get("license_status") not in ("approved", "public_domain", "verified_commercial"):
            raise ValueError(f"Unapproved rights: {row['id']}")
        sample = train[index]
        if sample["image"].shape != (6, 256, 256) or not sample["valid_mask"].any():
            raise ValueError(f"Invalid training image/valid mask: {row['id']}")
        positive = sample["target"].flatten(1).any(1)
        expected = "both" if positive.all() else "new_building" if positive[0] else "tree_removal" if positive[1] else "negative"
        is_real = row.get("source") == "NAIP-real-reviewed"
        if is_real:
            review = reviews.get(source_id)
            if review is None or review.get("label_status") != "reviewed":
                raise ValueError(f"Missing reviewed provenance: {row['id']}")
            if review.get("license_review_status") != "verified" or review.get("provenance_status") != "verified":
                raise ValueError(f"Missing license/provenance verification: {row['id']}")
            if any(row[k] != review[k] for k in ("pre", "post", "new_building", "tree_removal")):
                raise ValueError(f"Review files differ: {row['id']}")
            if expected == "negative":
                if review.get("review_status") != "accept" or review.get("partial_annotation") != "False":
                    raise ValueError(f"Negative must be fully reviewed: {row['id']}")
                if not sample["valid_mask"].all() or not sample["presence_valid"].all():
                    raise ValueError(f"Negative must be fully valid: {row['id']}")
                row["category"] = "real_negative"
                real_counts[row["region_id"]] += 1
            else:
                if not row.get("review_mask"):
                    raise ValueError(f"Partial positive lacks review mask: {row['id']}")
                row["category"] = expected
        else:
            if row.get("category") != expected:
                raise ValueError(f"Synthetic category/actual masks differ: {row['id']}")
            if expected == "negative" and row["pre"] != row["post"]:
                raise ValueError(f"Synthetic negative must use same scene: {row['id']}")
        categories[row["category"]] += 1
        rows.append(row)
    if sum(real_counts.values()) != 54:
        raise ValueError("Expected exactly 54 audited real negatives")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    with (root / "train.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "passed", "rows": len(rows), "categories": dict(categories),
        "real_negative_regions": dict(real_counts), "held_out_regions": sorted(held_regions),
        "source_id_overlap": 0, "path_overlap": 0, "exact_image_hash_overlap": 0,
        "real_negatives_fully_valid_zero_masks": 54,
        "reviewer_limitation": "Existing AI visual reviews, not independent human annotation.",
        "input_sha256": {str(p): digest(p) for p in (train_manifest, review_manifest, *validation_manifests)},
        "train_sha256": digest(root / "train.csv"),
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--reviews", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.train, a.reviews, a.validation, a.output), indent=2))
