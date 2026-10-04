"""A real-source fixture proves review decisions, exclusion and missing-class gates."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from terradelta.data.audit import audit_dataset
from terradelta.data.dataset import read_manifest
from terradelta.data.pairing import write_manifest


@pytest.fixture
def review(tmp_path):
    spec = importlib.util.spec_from_file_location("pilot_review", Path(__file__).parents[1] / "scripts/pilot_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, decisions = [], []
    for i in range(4):
        folder = tmp_path / f"tile_{i}"
        folder.mkdir()
        image = np.random.default_rng(i).integers(40, 200, (256, 256, 3), dtype=np.uint8)
        for role in ("pre", "post"):
            Image.fromarray(image if role == "pre" else np.roll(image, 1, axis=0)).save(folder / f"{role}.png")
        rows.append({"id": folder.name, "pre": str(folder / "pre.png"), "post": str(folder / "post.png"),
                     "source": "NAIP", "synthetic": False, "region_id": f"region_{i}",
                     "pre_source": "fixture_pre", "post_source": "fixture_post", "source_url": "fixture",
                     "bounds": json.dumps([500000+i*1000, 4000000, 500154+i*1000, 4000154]),
                     "crs": "EPSG:32618", "year_pre": 2018, "year_post": 2021,
                     "resolution_m": .6, "license_status": "unknown", "label_status": "candidate",
                     "audit_training_eligible": False})
        decisions.append({"id": folder.name, "status": "accept", "confidence": "HIGH",
                          "class": "no_change", "notes": "Fixture whole tile reviewed as unchanged",
                          "full_tile_reviewed": True, "board": "fixture_board.jpg",
                          "alignment_review": "Fixture known one-pixel translation"})
    write_manifest(rows, tmp_path / "candidates.csv")
    document = {"reviewer": "fixture_reviewer", "review_method": "fixture_full_tile_visual",
                "imagery_license_evidence": "fixture_provider_terms", "decisions": decisions}
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(document))
    return module, tmp_path, document, path


def test_only_explicit_high_reviews_export_and_background_only_is_not_ready(review):
    module, root, document, path = review
    document["decisions"] = document["decisions"][:3]
    document["decisions"][2].update(status="review", confidence="MEDIUM")
    path.write_text(json.dumps(document))
    result = module.apply_reviews(root / "candidates.csv", path, root / "processed")
    assert result["accepted_HIGH"] == 2 and result["unreviewed"] == 1
    exported = read_manifest(root / "processed/manifest.csv")
    assert all(set(np.unique(np.asarray(Image.open(Path(r["pre"]).parent / "mask.png")))) == {0} for r in exported)
    queued = read_manifest(root / "processed/review_queue.csv")
    assert {r["confidence"] for r in queued} == {"MEDIUM", "UNREVIEWED"}
    report = audit_dataset(root / "processed/manifest.csv", root / "audit",
                           train_manifest=root / "processed/train.csv", val_manifest=root / "processed/val.csv")
    assert report["readiness"] == "NOT READY"
    assert report["training_eligible_samples"] == 2
    assert sum("target class is missing" in b for b in report["blockers"]) == 2
    assert report["optimizer_steps"] == 0


@pytest.mark.parametrize("failure", ["scope", "confidence", "automatic_positive", "unknown_id"])
def test_incomplete_or_automatic_approval_fails_before_export(review, failure):
    module, root, document, path = review
    d = document["decisions"][0]
    if failure == "scope":
        d["full_tile_reviewed"] = False
    elif failure == "confidence":
        d["confidence"] = "MEDIUM"
    elif failure == "unknown_id":
        d["id"] = "missing"
    else:
        d.update({"class": "new_building", "label_license": "fixture_terms",
                  "masks": {"new_building": "proposed_new_building.png", "tree_removal": "final_tree.png"}})
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        module.apply_reviews(root / "candidates.csv", path, root / "processed")
    assert not (root / "processed").exists()
