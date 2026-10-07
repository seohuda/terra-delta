"""Small private-package fixtures; no production checkpoint or protected inputs."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

import nbformat
from PIL import Image
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = load_script("package_v3_satlas")


@pytest.fixture
def tiny_package(tmp_path, monkeypatch):
    notices = tmp_path / "repo"
    notices.mkdir()
    (notices / "LICENSE").write_text("MIT fixture\n")
    (notices / "THIRD_PARTY_NOTICES.md").write_text("Third party fixture\n")
    (notices / "third_party/satlas").mkdir(parents=True)
    (notices / "third_party/satlas/LICENSE").write_text("Satlas fixture\n")
    monkeypatch.setattr(package, "REPO_ROOT", notices)
    code = tmp_path / "code/terradelta"
    (code / "inference").mkdir(parents=True)
    (code / "utils").mkdir()
    for folder in (code, code / "inference", code / "utils"):
        (folder / "__init__.py").write_text("")
    (code / "utils/io.py").write_text((REPO / "src/terradelta/utils/io.py").read_text())
    # Exercise the real exported notebook contract and current-interpreter kernel.
    # Outbound connections/download hooks are forbidden inside this tiny CPU predictor.
    (code / "inference/satlas_v3.py").write_text('''import csv, socket, sys
from pathlib import Path
import torch
from PIL import Image
from terradelta.utils.io import write_prediction_csv

def predict_directory(input_dir, output_path, checkpoint, config, device):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    torch.hub.download_url_to_file = forbidden
    assert device == "cpu"
    assert sys.executable == config["expected_python"]
    assert Path(__file__).is_relative_to(Path(checkpoint).parents[1] / "code")
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    assert list(weights) == ["model_state_dict"]
    rows = []
    with (input_dir / "pairs.csv").open() as handle:
        for pair in csv.DictReader(handle):
            with Image.open(input_dir / "images" / pair["id"] / "pre.png") as image:
                assert image.size == (2, 2)
            rows.append({"id": pair["id"], "new_building": "EMPTY", "tree_removal": "EMPTY"})
    write_prediction_csv(output_path, rows)
    return rows
''')
    (code / "secret.pt").write_bytes(b"must not ship")
    checkpoint = tmp_path / "tiny.pt"
    state = {"linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    torch.save({"model_state_dict": state, "optimizer": {"step": 2}, "local_path": "/private"}, checkpoint)
    config = {"expected_python": sys.executable, "inference": {"batch_size": 1}}
    return checkpoint, config, code, tmp_path / "private.zip", state


def test_package_strips_only_metadata_and_includes_notices(tiny_package, tmp_path):
    checkpoint, config, code, output, state = tiny_package
    original = checkpoint.read_bytes()
    assert package.create_package(checkpoint, config, code, output) == output
    with zipfile.ZipFile(output) as archive:
        assert {"LICENSE", "THIRD_PARTY_NOTICES.md", "third_party/satlas/LICENSE"} <= set(archive.namelist())
        assert "assets/code/terradelta/secret.pt" not in archive.namelist()
        stripped = torch.load(io.BytesIO(archive.read("assets/model/model.pt")), weights_only=True)
        assert list(stripped) == ["model_state_dict"]
        assert torch.equal(stripped["model_state_dict"]["linear.weight"], state["linear.weight"])
        nbformat.validate(nbformat.reads(archive.read("predict.ipynb").decode(), as_version=4))
    assert checkpoint.read_bytes() == original
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        package.create_package(checkpoint, config, code, output)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob("terradelta-package-*"))


def test_publication_race_never_overwrites(tiny_package, monkeypatch):
    checkpoint, config, code, output, _ = tiny_package
    link = package.os.link

    def concurrent_writer(source, destination):
        destination.write_bytes(b"other writer")
        link(source, destination)

    monkeypatch.setattr(package.os, "link", concurrent_writer)
    with pytest.raises(FileExistsError):
        package.create_package(checkpoint, config, code, output)
    assert output.read_bytes() == b"other writer"


@pytest.mark.parametrize("name,mode", [
    ("../escape", 0), ("/absolute", 0), ("a/../../escape", 0),
    ("a\\escape", 0), ("C:/escape", 0), ("link", stat.S_IFLNK | 0o777),
])
def test_unsafe_zip_rejected_before_extraction(tmp_path, name, mode):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("valid.txt", "must not be extracted")
        info = zipfile.ZipInfo(name)
        info.external_attr = mode << 16
        archive.writestr(info, "target")
    destination = tmp_path / "clean"
    destination.mkdir()
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError):
        package.extract_package(archive, destination)
    assert not list(destination.iterdir())


def test_expansion_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "MAX_EXPANDED_BYTES", 4)
    archive_path = tmp_path / "large.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload", "x" * 5)
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="expansion"):
        package.extract_package(archive, tmp_path / "clean")


def test_two_pass_notebook_cpu_offline(tiny_package, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    checkpoint, config, code, output, _ = tiny_package
    package.create_package(checkpoint, config, code, output)
    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    manifest = tmp_path / "explicit.csv"
    manifest.write_text("id,pre,post\n0001,image.png,image.png\n")
    result = package.verify_cleanroom(output, manifest)
    assert result["byte_identical"]
    assert result["pass1_sha256"] == result["pass2_sha256"]
    assert result["zip_sha256"] == package.compute_sha256(output)


@pytest.mark.parametrize("anchor", ["__file__", "__vsc_ipynb_file__", "__notebook_path__",
                                    "AIF_SUBMISSION_DIR", "AIF_NOTEBOOK_PATH", "argv", "cwd"])
def test_original_submission_anchors(tmp_path, monkeypatch, anchor):
    root = tmp_path / "package"
    (root / "assets/model").mkdir(parents=True)
    (root / "assets/model/model.pt").touch()
    notebook = root / "predict.ipynb"
    notebook.touch()
    for name in ("AIF_SUBMISSION_DIR", "AIF_NOTEBOOK_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(root if anchor == "cwd" else tmp_path)
    monkeypatch.setattr(sys, "argv", [str(notebook)] if anchor == "argv" else [])
    namespace = {}
    if anchor.startswith("__"):
        namespace[anchor] = str(notebook)
    elif anchor.startswith("AIF"):
        monkeypatch.setenv(anchor, str(root if anchor == "AIF_SUBMISSION_DIR" else notebook))
    source = "".join(package.build_predict_notebook(Path("unused")).get("cells")[0]["source"])
    # Run the unchanged locator, before importing the deliberately separate API.
    exec(source[:source.index("ROOT = _submission_root()")], namespace)
    assert namespace["_submission_root"]() == root


def test_orchestrator_keeps_recipes_and_explicit_paths(tmp_path, monkeypatch):
    orchestrator = load_script("run_v3_experiments")
    calls = []
    monkeypatch.setattr(orchestrator, "run_cmd", lambda cmd, desc: calls.append(cmd))
    output = tmp_path / "output"
    (output / "S0").mkdir(parents=True)
    (output / "S0/checkpoint_step_200.pt").touch()

    def evaluate(cmd, check):
        calls.append(cmd)
        report = Path(cmd[cmd.index("--output") + 1])
        report.write_text(json.dumps({"legacy": {"score": 0, "bldg_tp": 0, "bldg_positives": 1,
            "tree_tp": 0, "tree_positives": 1, "no_change_fp": 0, "no_change_count": 1},
            "stress": {"score": 0, "bldg_tp": 0}, "gates": {"passed_all": False}}))

    monkeypatch.setattr(orchestrator.subprocess, "run", evaluate)
    monkeypatch.setattr(sys, "argv", ["run", "--output-dir", str(output), "--steps", "200",
                                     "--legacy-manifest", "legacy.csv", "--stress-manifest", "stress.csv"])
    orchestrator.main()
    assert [cmd[cmd.index("--recipe") + 1] for cmd in calls[:2]] == ["S1", "S2"]
    assert all(cmd[0] == sys.executable for cmd in calls)
    assert calls[2][-4:] == ["--legacy-manifest", "legacy.csv", "--stress-manifest", "stress.csv"]
