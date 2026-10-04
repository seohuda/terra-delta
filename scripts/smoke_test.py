#!/usr/bin/env python3
"""Small CPU-only end-to-end proof. Never trains, downloads data, or submits."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import yaml

from terradelta.data.dataset import ChangeDataset
from terradelta.data.pairing import write_manifest
from terradelta.data.split import split_manifest
from terradelta.inference.predictor import predict_directory
from terradelta.models.checkpoint import load_checkpoint
from terradelta.models.factory import build_model
from terradelta.submission import export_submission, make_submission_zip
from terradelta.training.search import search_thresholds
from terradelta.training.validation import validate_model
from terradelta.utils.io import atomic_json, load_config

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "baseline/original/03_illegal structure submission/assets/model/unet_r18_cd.pt")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/smoke")
    parser.add_argument("--execute-notebook", action="store_true", help="Also execute generated notebook using a local CPU Jupyter kernel")
    args = parser.parse_args()
    root, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if root.exists():
        parser.error(f"Output exists: {root}. Select a new directory to preserve previous evidence.")
    if not checkpoint.is_file():
        parser.error("Retrieve the official baseline per baseline/README.md first")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        folder = root / "input/images" / f"mock_{i:02}"
        folder.mkdir(parents=True)
        pre = rng.integers(70,160,(256,256,3),dtype=np.uint8)
        post = pre.copy()
        mask = np.zeros((256,256),np.uint8)
        if i in (0,1):
            post[60:100,80:120] = [180,190,170]
            mask[60:100,80:120] = 1 if i == 0 else 2
        for name, image in (("pre",pre),("post",post),("new_building",(mask==1).astype(np.uint8)*255),
                            ("tree_removal",(mask==2).astype(np.uint8)*255)):
            Image.fromarray(image).save(folder / f"{name}.png")
        rows.append({"id":folder.name, **{k:str(folder / f"{k}.png") for k in ("pre","post","new_building","tree_removal")},
                     "region_id":f"mock_region_{i}", "state":"mock", "source":"synthetic_mock",
                     "year_pre":2020,"year_post":2022,"license_status":"commercial_ok","label_status":"reviewed"})
    (root / "input/pairs.csv").write_text("id\n"+"\n".join(r["id"] for r in rows)+"\n")
    write_manifest(rows,root / "manifest.csv")
    parts = split_manifest(rows, strategy="region",val_fraction=.25,seed=0)
    for name, values in parts.items():
        write_manifest(values, root / f"{name}.csv")
    train_config = load_config(REPO / "configs/train_short.yaml")
    train_config["data"]["manifest"] = str(root / "manifest.csv")
    train_config["training"].update(device="cpu",init_checkpoint=str(checkpoint),output_dir=str(root / "unused_training"))
    config_path = root / "smoke_train.yaml"
    config_path.write_text(yaml.safe_dump(train_config,sort_keys=False))
    # Only this exact dry-run command is permitted. The CLI does not construct an optimizer.
    command = [sys.executable,str(REPO / "scripts/train.py"),"--config",str(config_path),"--dry-run","--device","cpu"]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    run = subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True,check=True,timeout=60)
    plan = json.loads(run.stdout)
    assert plan["optimizer_steps_executed"] == 0 and not plan["optimizer_constructed"]
    atomic_json(root / "training_dry_run.json",plan)
    inference = load_config(REPO / "configs/inference_baseline.yaml")
    predictions = predict_directory(root / "input",root / "prediction.csv",checkpoint,inference,
                                    probability_dir=root / "probabilities")
    model = build_model().eval()
    load_checkpoint(checkpoint,model)
    validation = validate_model(model,ChangeDataset(root / "manifest.csv"),inference,"cpu")
    atomic_json(root / "mock_validation.json",validation)
    grid = {name:dict(threshold=[.5,.6],min_area=[20,30],min_pos_area=[20],simplify_px=[0,.5])
            for name in ("new_building","tree_removal")}
    search = search_thresholds(root / "probabilities",root / "manifest.csv",inference,
        objective="score",grid=grid,method="staged",output_csv=root / "threshold_search.csv",best_yaml=root / "best_postprocess.yaml")
    export = export_submission(checkpoint,REPO / "configs/inference_baseline.yaml",root / "submission_export")
    archive = make_submission_zip(export,root / "submission.zip")
    notebook_executed = False
    if args.execute_notebook:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        spec_root = root / "kernels/terradelta-smoke"
        spec_root.mkdir(parents=True)
        spec = {"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
            "display_name":"TerraDelta CPU smoke","language":"python",
            "env":{**{k:env[k] for k in ("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                   "AIF_INPUT_DIR":str(root / "input"),"AIF_PREDICTION_PATH":str(root / "notebook_prediction.csv")}}
        (spec_root / "kernel.json").write_text(json.dumps(spec))
        manager = KernelManager(kernel_name="terradelta-smoke", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_root.parent)]))
        nb = nbformat.read(export / "predict.ipynb",as_version=4)
        # Test-only prelude blocks outbound Python HTTP/socket operations.
        nb.cells[1].source = '''import socket
import torch
def no_network(*args, **kwargs):
    raise AssertionError("Offline smoke attempted network access")
socket.socket.connect = no_network
socket.create_connection = no_network
torch.hub.download_url_to_file = no_network
''' + nb.cells[1].source
        client = NotebookClient(nb,km=manager,timeout=60,resources={"metadata":{"path":str(export)}})
        client.execute()
        nbformat.write(nb, root / "executed_predict.ipynb")
        assert (root / "notebook_prediction.csv").read_bytes() == (root / "prediction.csv").read_bytes()
        notebook_executed = True
    report = {"mock_only":True,"samples":len(rows),"prediction_rows":len(predictions),
              "optimizer_steps_executed":0,"training_executed":False,"notebook_executed":notebook_executed,
              "zip":str(archive),"zip_bytes":archive.stat().st_size,"threshold_trials":len(search["results"]),
              "score_note":"All validation/search numbers use invented mock labels, not model quality evidence."}
    atomic_json(root / "report.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
