"""Execute only the extracted v2 notebook with network and training blocked."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image

from terradelta.data.dataset_v2 import read_v2_manifest
from terradelta.inference.v2_calibration import digest
from terradelta.utils.io import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--stress", required=True)
    args = parser.parse_args()
    base = Path(args.package_root).resolve()
    archive = base / "terradelta-v2-debug.zip"
    selection = json.loads((base / "final-v2-selection.json").read_text())
    clean = base / "cleanroom"
    clean.mkdir(exist_ok=False)
    members = []
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Invalid ZIP CRC")
        for info in z.infolist():
            n = info.filename
            if (
                n.startswith("/")
                or ".." in Path(n).parts
                or any(
                    s in n
                    for s in ("calibration", "/training/", "/data/", "__pycache__", ".pyc", ".venv", ".git")
                )
            ):
                raise ValueError(f"Forbidden ZIP member {n}")
            members.append(
                {"path": n, "bytes": info.file_size, "sha256": hashlib.sha256(z.read(n)).hexdigest()}
            )
        z.extractall(clean)
    if digest(clean / "assets/model/model.pt") != selection["checkpoint_sha256"]:
        raise ValueError("ZIP checkpoint differs from selected checkpoint")
    if digest(clean / "assets/config.yaml") != selection["config_sha256"]:
        raise ValueError("ZIP config differs from selected config")
    input_dir = base / "검증 입력"
    selected_rows = read_v2_manifest(args.legacy)
    stress = read_v2_manifest(args.stress)
    for kind in ("building", "tree", "both", "negative"):
        matched = []
        for row in stress:
            positives = []
            for name in ("new_building", "tree_removal"):
                value = row[name]
                if value.lower() == "absent":
                    positives.append(False)
                else:
                    with Image.open(value) as mask:
                        positives.append(bool(np.asarray(mask.convert("L")).any()))
            building, tree = positives
            label = (
                "both" if building and tree else "building" if building else "tree" if tree else "negative"
            )
            if label == kind:
                matched.append(row)
        selected_rows.extend(matched[:2])
    ids, input_hashes = [], {}
    for row in selected_rows:
        identifier = row["id"]
        if identifier in ids:
            raise ValueError("Duplicate test ID")
        ids.append(identifier)
        folder = input_dir / "images" / identifier
        folder.mkdir(parents=True)
        for name in ("pre", "post"):
            source = Path(row[name])
            input_hashes[str(source)] = digest(source)
            shutil.copyfile(source, folder / (name + ".png"))
    for identifier, level in [("0001", 0), ("한글_0002", 128)]:
        ids.append(identifier)
        folder = input_dir / "images" / identifier
        folder.mkdir(parents=True)
        for name in ("pre", "post"):
            Image.fromarray(np.full((256, 256, 3), level, np.uint8)).save(folder / (name + ".png"))
    with (input_dir / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id"])
        writer.writeheader()
        writer.writerows({"id": i} for i in ids)
    notebook = json.loads((clean / "predict.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    if any(line.lstrip().startswith(("!", "%")) for source in cells for line in source.splitlines()):
        raise ValueError("Install/submit notebook commands forbidden")
    prelude = """import socket,torch,sys,json,time,csv,math
from pathlib import Path
started=time.monotonic()
def forbidden(*a,**kw): raise AssertionError("Network, optimizer and backward forbidden")
socket.socket.connect=forbidden
socket.socket.connect_ex=forbidden
socket.create_connection=forbidden
torch.hub.download_url_to_file=forbidden
torch.optim.Optimizer.__init__=forbidden
torch.Tensor.backward=forbidden
torch.autograd.backward=forbidden
torch.cuda.is_available=lambda:False
"""
    suffix = """
from terradelta.inference.v2 import output_row
from terradelta.utils.io import write_prediction_csv
from shapely.geometry import Polygon
import numpy as np
assert DEVICE=='cpu'
for name,module in list(sys.modules.items()):
    if name=='terradelta' or name.startswith('terradelta.'):
        assert Path(module.__file__).resolve().is_relative_to(ROOT/'assets/code'),name
def validate(path,ids):
    with open(path,newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['id','new_building','tree_removal'];rows=list(reader)
    assert [r['id'] for r in rows]==ids
    for row in rows:
        for name in ('new_building','tree_removal'):
            for exterior in json.loads(row[name]) if row[name] else []:
                assert len(exterior)>=3
                assert all(len(point)==2 and all(math.isfinite(v) and 0<=v<=256 for v in point) for point in exterior)
                poly=Polygon(exterior);assert poly.is_valid and poly.area>0
    return rows
ids=[r['id'] for r in csv.DictReader(open(INPUT_DIR/'pairs.csv'))]
actual=validate(PREDICTION_PATH,ids)
p=np.zeros((2,256,256),np.float32);q=np.ones(2,np.float32)
empty=output_row('empty',p,q,CONFIG);assert empty['new_building']==empty['tree_removal']==''
p[:,20:60,20:60]=1
both=output_row('both',p,q,CONFIG)
assert both['new_building']==both['tree_removal']!=''
gated=output_row('gated',p,np.zeros(2,np.float32),{'postprocess':{'presence_threshold':.5,'pixel_threshold':0}})
assert gated['new_building']==gated['tree_removal']==''
write_prediction_csv(ROOT.parent/'contract.csv',[empty,both,gated])
validate(ROOT.parent/'contract.csv',['empty','both','gated'])
report={'status':'passed','device':DEVICE,'rows':len(actual),'ids_preserved':True,
        'columns_valid':True,'finite_bounded_valid_polygons':True,'empty_cells':True,
        'independent_overlapping_heads':True,'presence_before_pixel':True,'network_blocked':True,
        'optimizer_and_backward_blocked':True,'repository_imports':False,
        'elapsed_seconds':time.monotonic()-started}
print(json.dumps(report))
"""
    environment = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", AIF_INPUT_DIR=str(input_dir))
    reports, outputs = [], []
    for index in range(2):
        path = base / f"cleanroom-prediction-{index}.csv"
        started = time.monotonic()
        run = subprocess.run(
            [sys.executable, "-I", "-c", prelude + "\n".join(cells) + suffix],
            cwd=clean,
            env=dict(environment, AIF_PREDICTION_PATH=str(path)),
            text=True,
            capture_output=True,
            timeout=600,
        )
        (base / f"cleanroom-{index}-stdout.log").write_text(run.stdout)
        (base / f"cleanroom-{index}-stderr.log").write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
        report = json.loads(run.stdout.strip().splitlines()[-1])
        report["process_seconds"] = time.monotonic() - started
        reports.append(report)
        outputs.append(path.read_bytes())
    if outputs[0] != outputs[1]:
        raise ValueError("Repeat inference is not byte deterministic")
    if any(digest(Path(path)) != expected for path, expected in input_hashes.items()):
        raise ValueError("Source validation input changed")
    manifest = {
        "status": "cleanroom_passed",
        "artifact": archive.name,
        "artifact_bytes": archive.stat().st_size,
        "artifact_sha256": digest(archive),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "config_sha256": selection["config_sha256"],
        "selected_step": selection["step"],
        "files": members,
        "cleanroom": reports[0],
        "repeat_cleanroom": reports[1],
        "deterministic_repeat": True,
        "input_hashes_unchanged": True,
        "requirements": (clean / "requirements.txt").read_text().splitlines(),
        "submission_count": 0,
        "main_count": 0,
        "training_steps_executed": 0,
    }
    atomic_json(base / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
