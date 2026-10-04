"""Run exported notebook cells in a fresh CPU process with outbound sockets blocked."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import zipfile

import nbformat
import numpy as np
from PIL import Image
import pytest
import torch

from terradelta.inference.predictor import predict_directory, discover_pairs
from terradelta.submission import export_submission, make_submission_zip
from terradelta.utils.io import load_config, write_prediction_csv

REPO = Path(__file__).resolve().parents[1]


def create_input(root):
    path = root / "wrapped"
    (path / "images/0001").mkdir(parents=True)
    (path / "pairs.csv").write_text("\ufeffid\n0001\n", encoding="utf-8")
    rng = np.random.default_rng(1)
    for name in ("pre", "post"):
        Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8)).save(path / f"images/0001/{name}.png")
    return root


@pytest.mark.parametrize("config_name", ["inference_baseline.yaml", "inference_conservative.yaml"])
def test_zip_root_and_offline_cells(tmp_path, official_checkpoint, config_name):
    config_path = REPO / "configs" / config_name
    export = export_submission(official_checkpoint, config_path, tmp_path / "export")
    assert hashlib.sha256((export / "assets/model/model.pt").read_bytes()).digest() == hashlib.sha256(official_checkpoint.read_bytes()).digest()
    archive = make_submission_zip(export, tmp_path / "submission.zip")
    extracted = tmp_path / "isolated"
    with zipfile.ZipFile(archive) as z:
        assert {"predict.ipynb", "requirements.txt", "LICENSE", "NOTICE", "assets/model/model.pt", "assets/config.yaml"} <= set(z.namelist())
        assert not any(n.startswith("export/") for n in z.namelist())
        z.extractall(extracted)
    notebook = nbformat.read(extracted / "predict.ipynb", as_version=4)
    nbformat.validate(notebook)
    input_dir = create_input(tmp_path / "input")
    output = tmp_path / "actual.csv"
    # Executes exactly all notebook code cells; submit/install magic is forbidden.
    codes = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert all(not line.lstrip().startswith(("!", "%")) for source in codes for line in source.splitlines())
    assert not any("aifactory submit" in source for source in codes)
    prelude = '''import socket, torch
from pathlib import Path
torch.set_num_threads(2)
def forbidden(*args, **kwargs):
    raise AssertionError("Network access forbidden during inference")
socket.socket.connect = forbidden
socket.create_connection = forbidden
torch.hub.download_url_to_file = forbidden
'''
    suffix = '\nimport terradelta\nassert Path(terradelta.__file__).resolve().is_relative_to(Path.cwd() / "assets/code")\n'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", AIF_INPUT_DIR=str(input_dir), AIF_PREDICTION_PATH=str(output))
    run = subprocess.run([str(Path(os.sys.executable).absolute()), "-c", prelude + "\n".join(codes) + suffix],
                         cwd=extracted, env=environment, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    expected = tmp_path / "expected.csv"
    predict_directory(input_dir, expected, official_checkpoint, load_config(config_path), probability_dir=tmp_path / "prob")
    assert output.read_bytes() == expected.read_bytes()
    probabilities = np.load(tmp_path / "prob/0001.npy", allow_pickle=False)
    assert probabilities.shape == (3,256,256) and probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities.sum(0), 1, atol=1e-6)
    with output.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["id"] == "0001"


def test_csv_empty_and_json_roundtrip(tmp_path):
    rows = [{"id":"001", "new_building":"", "tree_removal":"[[[0,0],[5,0],[5,5],[0,5]]]"}]
    path = tmp_path / "nested/result.csv"
    write_prediction_csv(path, rows)
    with path.open(newline="") as f:
        assert list(csv.DictReader(f)) == rows
    with pytest.raises(ValueError, match="Duplicate"):
        write_prediction_csv(path, rows * 2)


def test_safe_input_zip_and_traversal(tmp_path):
    root = create_input(tmp_path / "data")
    zip_input = tmp_path / "zipped"
    zip_input.mkdir()
    with zipfile.ZipFile(zip_input / "input.zip", "w") as z:
        for file in root.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(root).as_posix())
    resolved, ids = discover_pairs(zip_input)
    assert ids == ["0001"] and (resolved / "images/0001/pre.png").exists()
    bad = tmp_path / "bad"
    bad.mkdir()
    with zipfile.ZipFile(bad / "evil.zip", "w") as z:
        z.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        discover_pairs(bad)


def test_export_rejects_incompatible_weights(tmp_path):
    checkpoint = tmp_path / "incompatible.pt"
    torch.save({"state_dict":{"invalid":torch.zeros(1)}, "in_channels":3}, checkpoint)
    with pytest.raises(ValueError):
        export_submission(checkpoint, REPO / "configs/inference_baseline.yaml", tmp_path / "out")


def test_export_strips_training_state(tmp_path, official_checkpoint):
    payload = torch.load(official_checkpoint,weights_only=True,map_location="cpu")
    payload.update(optimizer={"example":torch.zeros(1)}, training_format_version=1,
                   config={"data":{"manifest":"private/local/path.csv"}},rng={"test":1})
    source = tmp_path / "training.pt"
    torch.save(payload,source)
    output = export_submission(source,REPO / "configs/inference_baseline.yaml",tmp_path / "export")
    exported = torch.load(output / "assets/model/model.pt",weights_only=True,map_location="cpu")
    assert set(exported)=={"state_dict","classes","in_channels","encoder","steps","seed"}
    for key,value in payload["state_dict"].items():
        torch.testing.assert_close(value,exported["state_dict"][key])
