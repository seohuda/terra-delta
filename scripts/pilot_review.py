#!/usr/bin/env python3
"""Apply explicit full-tile visual decisions to an offline real-only pilot.

Unreviewed/medium/low samples retain their candidate status. No spectral or
static footprint proposal becomes ground truth by default.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from terradelta.data.dataset import ChangeDataset, MASK_CLASSES, read_manifest
from terradelta.data.pairing import write_manifest
from terradelta.data.split import split_manifest


def apply_reviews(manifest, decisions_path, output):
    rows = read_manifest(manifest)
    decisions_path, output = Path(decisions_path).resolve(), Path(output).resolve()
    document = json.loads(decisions_path.read_text())
    if not document.get("reviewer") or not document.get("review_method"):
        raise ValueError("Record the reviewer and actual visual review method")
    decisions = document["decisions"]
    ids = [d["id"] for d in decisions]
    if len(set(ids)) != len(ids) or not set(ids) <= {r["id"] for r in rows}:
        raise ValueError("Decisions contain duplicate or unknown IDs")
    index = {d["id"]: d for d in decisions}
    accepted, queue, masks = [], [], {}
    # Validate every decision before producing any files.
    for row in rows:
        d = index.get(row["id"])
        if not d:
            queue.append({**row, "review_status": "pending", "confidence": "UNREVIEWED"})
            continue
        if d.get("status") not in {"accept", "review", "reject"} or d.get("confidence") not in {"HIGH", "MEDIUM", "LOW"} or not d.get("notes"):
            raise ValueError("Every decision needs status, confidence and concrete notes")
        if (d["status"] == "accept") != (d["confidence"] == "HIGH"):
            raise ValueError("Only explicit HIGH decisions may enter the pilot")
        metadata = {**row, "confidence": d["confidence"], "review_status": d["status"],
                    "reviewer": document["reviewer"], "review_method": document["review_method"],
                    "review_notes": d["notes"], "review_board": d.get("board", "")}
        if d["status"] != "accept":
            queue.append(metadata)
            continue
        if str(row.get("synthetic", "")).lower() in {"true", "1"} or row.get("source") != "NAIP":
            raise ValueError("This pilot accepts real NAIP pairs only")
        if d.get("full_tile_reviewed") is not True or not d.get("board"):
            raise ValueError("Acceptance needs full-tile review and visual evidence")
        if not d.get("alignment_review"):
            raise ValueError("Acceptance requires an alignment decision")
        evidence = document.get("imagery_license_evidence", "")
        if not evidence:
            raise ValueError("Accepted NAIP data need documented imagery license evidence")
        images = [np.asarray(Image.open(row[k])) for k in ("pre", "post")]
        if any(a.shape != (256, 256, 3) or a.dtype != np.uint8 for a in images):
            raise ValueError("Accepted pilot imagery must be 256x256 uint8 RGB")
        if d.get("class") == "no_change":
            if d.get("masks"):
                raise ValueError("No-change decisions cannot carry positive masks")
            mask = np.zeros((256, 256), dtype=np.uint8)
        else:
            # Supply separately reviewed final masks; never infer absence.
            if set(d.get("masks", {})) != set(MASK_CLASSES):
                raise ValueError("Positive decisions require both explicitly reviewed masks")
            if not d.get("label_license"):
                raise ValueError("Positive annotations require their own documented license")
            final = {key: str((decisions_path.parent / value).resolve()) for key, value in d["masks"].items()}
            if any(Path(v).name.startswith(("proposed_", "hansen_", "building_footprint_reference")) for v in final.values()):
                raise ValueError("Candidate evidence must be refined into final reviewed masks")
            mask = ChangeDataset([{**row, **final}])[0]["mask"].numpy().astype(np.uint8)
            if not mask.any():
                raise ValueError("Positive decision has empty masks")
        masks[row["id"]] = mask
        for key in ("audit_training_eligible", "audit_status", "audit_reason"):
            metadata.pop(key, None)
        metadata.update(label_status="reviewed", label_source=document["reviewer"],
                        mask_source=document["reviewer"], license_status="public_domain",
                        license_source=evidence, imagery_license="USDA NAIP public domain",
                        label_license=d.get("label_license", "NAIP public-domain observations; empty absence masks"),
                        license_review_status="verified", provenance_status="verified",
                        synthetic=False, alignment_review=d["alignment_review"],
                        negative_type=d.get("negative_type", ""))
        accepted.append(metadata)
    if output.exists():
        raise FileExistsError("Use a fresh output; previous annotations remain preserved")
    if not accepted:
        raise ValueError("No explicit HIGH decisions to export")
    partitions = split_manifest(accepted, strategy="region", seed=0)
    output.mkdir(parents=True)
    for row in accepted:
        folder = output / row["id"]
        folder.mkdir()
        for key in ("pre", "post"):
            shutil.copyfile(row[key], folder / f"{key}.png")
            row[key] = str(folder / f"{key}.png")
        mask = masks[row["id"]]
        Image.fromarray(mask).save(folder / "mask.png")
        for label, key in enumerate(MASK_CLASSES, 1):
            Image.fromarray((mask == label).astype(np.uint8) * 255).save(folder / f"{key}.png")
            row[key] = str(folder / f"{key}.png")
            row[key + "_present"] = bool((mask == label).any())
        (folder / "metadata.json").write_text(json.dumps(row, indent=2))
    write_manifest(accepted, output / "manifest.csv")
    # split_manifest retains each row object; paths above now point to the export.
    for name, subset in partitions.items():
        write_manifest(subset, output / f"{name}.csv")
    write_manifest(queue, output / "review_queue.csv")
    shutil.copyfile(decisions_path, output / "review_decisions.json")
    result = {"candidate_pairs": len(rows), "accepted_HIGH": len(accepted),
              "reviewed_confidence": dict(Counter(d["confidence"] for d in decisions)),
              "unreviewed": len(rows) - len(decisions),
              "rejected": sum(d["status"] == "reject" for d in decisions),
              "train": len(partitions["train"]), "val": len(partitions["val"]),
              "training_executed": False}
    (output / "review_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(apply_reviews(args.candidates, args.decisions, args.output)))
