#!/usr/bin/env python3
"""Offline pre-training audit; CPU existing-weight inference, no fitting or downloads."""
import argparse
import json
from pathlib import Path
import socket

from terradelta.data.audit import audit_dataset, inventory
from terradelta.utils.io import atomic_json, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--scan-root", default=".")
    parser.add_argument("--output", default="outputs/data_audit")
    parser.add_argument("--checkpoint", help="Existing local official weights; never downloaded")
    parser.add_argument("--config", default="configs/inference_baseline.yaml")
    args = parser.parse_args(argv)

    def offline(*unused, **kwargs):
        raise RuntimeError("Dataset audit forbids network access")
    socket.socket.connect = offline
    socket.create_connection = offline
    import torch
    torch.set_num_threads(2)
    report = audit_dataset(args.manifest, args.output, train_manifest=args.train_manifest,
                           val_manifest=args.val_manifest, checkpoint=args.checkpoint,
                           config=load_config(args.config))
    scanned = inventory(args.scan_root)
    atomic_json(Path(args.output) / "inventory.json", scanned)
    report["inventory"] = scanned
    # Include the report itself; two digit-count changes are enough to stabilize.
    for _ in range(3):
        report["audit_output_bytes"] = sum(p.stat().st_size for p in Path(args.output).rglob("*") if p.is_file())
        atomic_json(Path(args.output) / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
