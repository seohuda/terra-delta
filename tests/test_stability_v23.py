"""Frozen forward/matching/package contracts; no training or optimizer steps."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
from shapely.geometry import box
import torch
import yaml

from terradelta.inference.stability_v23 import (
    CLASSES, RULES, TRANSFORMS, V23Predictor, apply_stability, component_features,
    match_components, radiometric_pair, transform, tta_probabilities, validate_stability,
)
from terradelta.inference.v2 import V2Predictor, output_row
from terradelta.inference.verifier import FEATURES, FROZEN_V2_SHA256
from terradelta.submission import export_submission, make_notebook, make_submission_zip
from terradelta.utils.io import write_prediction_csv


def config():
    return {"postprocess": {"presence_threshold": None, "pixel_threshold": .5,
            "min_area": 30, "min_pos_area": 20, "simplify_px": .5, "ndigits": 2}}


def gate(rule="A", **extra):
    return {"enabled": True, "classes": {c: copy.deepcopy(RULES[rule]) for c in CLASSES}, **extra}


def maps():
    p = np.zeros((4, 2, 256, 256), np.float32)
    p[:, :, 20:35, 40:60] = .9
    p[0, :, 140:152, 170:183] = .8
    return p, np.ones((4, 2), np.float32)


@pytest.mark.parametrize("name", TRANSFORMS)
@pytest.mark.parametrize("tensor", [False, True])
def test_transform_and_inverse_preserve_coordinates_and_pair(name, tensor):
    value = np.arange(2 * 6 * 11 * 13).reshape(2, 6, 11, 13)
    original = torch.from_numpy(value.copy()) if tensor else value
    actual = transform(transform(original, name), name)
    assert np.array_equal(np.asarray(actual), value)
    transformed = np.asarray(transform(original, name))
    if name == "hflip":
        assert transformed[0, 0, 0, 0] == value[0, 0, 0, -1]
    elif name == "vflip":
        assert transformed[0, 0, 0, 0] == value[0, 0, -1, 0]
    elif name == "rot180":
        assert transformed[0, 0, 0, 0] == value[0, 0, -1, -1]
    else:
        assert np.array_equal(transformed, value)
    assert np.array_equal(np.asarray(original), value)


def test_unknown_transform_rejected():
    with pytest.raises(ValueError):
        transform(np.zeros((2, 2)), "rot90")


def test_matching_fixed_iou_centroid_rule_empty_and_one_to_one():
    identity = [box(0, 0, 10, 10), box(30, 30, 40, 40)]
    other = [box(31, 30, 41, 40), box(1, 0, 11, 10)]
    matches = match_components(identity, other)
    assert matches[0]["index"] == 1 and matches[1]["index"] == 0
    assert matches[0]["iou"] == pytest.approx(90 / 110)
    assert matches[0]["centroid_drift"] == 1
    # Tiny centroid shifts can qualify despite low IoU, but extreme area ratios cannot.
    assert 0 in match_components([box(0, 0, 8, 8)], [box(1, 1, 5, 5)])
    assert not match_components([box(0, 0, 8, 8)], [box(0, 0, 1, 1)])
    assert not match_components(identity, [])
    assert not match_components([], other)
    assert len(match_components([identity[0], identity[0]], [other[1]])) == 1
    assert match_components([identity[0], identity[0]], [other[1]]) == match_components(
        [identity[0], identity[0]], [other[1]])


def test_features_and_only_rejected_components_removed_without_repolygonizing():
    p, q = maps()
    row = output_row("0001", p[0], q[0], config())
    source = copy.deepcopy(p)
    features = component_features(p, q, config())
    stable = next(r for r in features[CLASSES[0]] if r["persistence_count"] == 4)
    unstable = next(r for r in features[CLASSES[0]] if r["persistence_count"] == 1)
    assert stable["component_area"] == 300
    assert stable["original_mean_probability"] == pytest.approx(.9)
    assert stable["original_max_probability"] == pytest.approx(.9)
    assert stable["mean_matching_iou"] == stable["minimum_matching_iou"] == 1
    assert stable["area_coefficient_of_variation"] == stable["centroid_drift"] == 0
    assert stable["consensus_fraction"] == 1
    assert unstable["mean_matching_iou"] == 0 and unstable["persistence_fraction"] == .25
    result = apply_stability(row, features, gate())
    assert json.loads(result[CLASSES[0]]) == [stable["polygon"]]
    assert stable["polygon"] in json.loads(row[CLASSES[0]])
    assert np.array_equal(p, source)
    assert result == apply_stability(row, component_features(p, q, config()), gate())


def test_identity_shapes_holes_and_diagonal_components_match_reference():
    p = np.zeros((4, 2, 256, 256), np.float32)
    p[:, :, 20:50, 20:50] = 1
    p[:, :, 25:30, 25:30] = 0
    p[:, :, 50:60, 50:60] = 1  # diagonal-only contact remains separate.
    q = np.ones((4, 2))
    row = output_row("holes", p[0], q[0], config())
    features = component_features(p, q, config())
    assert len(features[CLASSES[0]]) == 2
    assert sorted(f["component_area"] for f in features[CLASSES[0]]) == [100, 875]
    assert apply_stability(row, features, gate("B")) == row


def test_class_isolation_presence_veto_and_empty_prediction():
    p, q = maps()
    p[:, 1] = p[0, 1]  # Both tree components are stable; building loses one.
    row = output_row("classes", p[0], q[0], config())
    result = apply_stability(row, component_features(p, q, config()), gate())
    assert len(json.loads(result[CLASSES[0]])) == 1
    assert result[CLASSES[1]] == row[CLASSES[1]]
    p[:, 1] = 0
    empty = output_row("empty", np.zeros_like(p[0]), q[0], config())
    assert apply_stability(empty, {}, gate()) == empty
    settings = config()
    settings["postprocess"]["presence_threshold"] = .5
    q[1:, 0] = .1
    assert component_features(p, q, settings)[CLASSES[0]][0]["persistence_count"] == 1


class EquivariantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        pixels = torch.cat((image[:, :1], image[:, 1:2]), 1) + self.anchor
        return {"segmentation": pixels, "presence": image.new_full((len(image), 2), 4.)}


def verifier():
    return {"version": 1, "features": list(FEATURES), "checkpoint_sha256": FROZEN_V2_SHA256,
            "mean": np.zeros((2, len(FEATURES))).tolist(), "scale": np.ones((2, len(FEATURES))).tolist(),
            "weight": np.zeros((2, len(FEATURES))).tolist(), "bias": [-3., 3.],
            "threshold": [.5, .5], "pixel_thresholds": [.5, .5]}


def test_disabled_is_v22_byte_identical_and_requires_only_one_forward(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Optimizer/backward is prohibited")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    settings = config()
    settings.update(verifier=verifier(), stability={"enabled": False})
    image = torch.full((1, 6, 256, 256), -4.)
    image[:, :2, 10:30, 30:60] = 4.
    model = EquivariantModel()
    predictor = V23Predictor(None, settings, model=model)
    p, q = predictor.probabilities(image)
    reference = V2Predictor(None, settings, model=EquivariantModel()).output_rows(["001"], image, p, q)
    actual = predictor.output_rows(["001"], image, p, q)
    write_prediction_csv(tmp_path / "v22.csv", reference)
    write_prediction_csv(tmp_path / "v23-disabled.csv", actual)
    assert (tmp_path / "v22.csv").read_bytes() == (tmp_path / "v23-disabled.csv").read_bytes()
    assert model.calls == 1
    assert all(param.grad is None and not param.requires_grad for param in model.parameters())


def test_v22_verifier_veto_cannot_be_resurrected_by_stability():
    settings = config()
    settings.update(verifier=verifier(), stability=gate("B"))
    image = torch.full((1, 6, 256, 256), -4.)
    image[:, :2, 10:30, 30:60] = 4.
    model = EquivariantModel()
    predictor = V23Predictor(None, settings, model=model)
    p, q = predictor.probabilities(image)
    original = V2Predictor.output_rows(predictor, ["001"], image, p, q)[0]
    actual = predictor.output_rows(["001"], image, p, q)[0]
    assert actual[CLASSES[0]] == "" and actual[CLASSES[1]] == original[CLASSES[1]] != ""
    assert model.calls == 4


def test_restored_tta_probabilities_equal_identity_for_equivariant_model():
    image = torch.full((1, 6, 256, 256), -4.)
    image[:, 0, 11:40, 60:120] = 4.
    image[:, 1, 180:220, 7:35] = 4.
    model = EquivariantModel()
    predictor = V2Predictor(None, config(), model=model)
    p, q = tta_probabilities(predictor, image)
    assert model.calls == 4
    assert np.array_equal(p, np.broadcast_to(p[:1], p.shape))
    assert np.array_equal(q, np.broadcast_to(q[:1], q.shape))


def test_appearance_is_auxiliary_only_and_never_replaces_identity_polygons():
    p, q = maps()
    aux = p[0].copy()
    aux[0] = 0
    features = component_features(p, q, config(), (aux, q[0]))
    row = output_row("aux", p[0], q[0], config())
    result = apply_stability(row, features, gate(appearance_consistency=True))
    assert result[CLASSES[0]] == ""
    stable_tree = next(f["polygon"] for f in features[CLASSES[1]] if f["persistence_count"] == 4)
    assert json.loads(result[CLASSES[1]]) == [stable_tree]
    image = torch.zeros(2, 6, 256, 256)
    fixed = radiometric_pair(image)
    assert torch.isfinite(fixed).all() and torch.equal(image[:, 3:], fixed[:, 3:])
    assert torch.equal(image, torch.zeros_like(image))


@pytest.mark.parametrize("bad", [
    {"enabled": "yes"}, {"classes": {"other": RULES["A"]}},
    {"classes": {CLASSES[0]: {"persistence_min": True, "mean_iou_min": .2}}},
    {"classes": {CLASSES[0]: {"persistence_min": 5, "mean_iou_min": .2}}},
    {"classes": {CLASSES[0]: {"persistence_min": 3, "mean_iou_min": float('nan')}}},
    {"appearance_consistency": 1}, {"matching_threshold": .9},
])
def test_bad_gate_configuration_rejected(bad):
    with pytest.raises(ValueError):
        validate_stability(bad)


def test_invalid_maps_mismatched_identity_and_missing_appearance_fail():
    p, q = maps()
    row = output_row("invalid", p[0], q[0], config())
    features = component_features(p, q, config())
    changed = dict(row, new_building="[[[0,0],[1,0],[1,1]]]")
    with pytest.raises(ValueError, match="identity polygon"):
        apply_stability(changed, features, gate())
    with pytest.raises(ValueError, match="auxiliary inference"):
        apply_stability(row, features, gate(appearance_consistency=True))
    p[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        component_features(p, q, config())


def test_notebook_routes_to_new_module_only_for_stability():
    notebook = make_notebook("siamese_v2", stability=True)
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    assert "from terradelta.inference.stability_v23 import predict_directory" in source
    assert "stability_v23" not in json.dumps(make_notebook("siamese_v2"))
    assert "http://" not in source and "https://" not in source and "api_key" not in source.lower()
    assert not any(line.lstrip().startswith(('!', '%')) for line in source.splitlines())
    with pytest.raises(ValueError, match="Siamese"):
        make_notebook("baseline", stability=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_offline_frozen_v23_package(tmp_path, enabled):
    """Executed on EC2 with retained exact weights; never downloads them locally."""
    checkpoint = os.environ.get("TERRADELTA_FROZEN_V2_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Exact frozen weights stay on EC2; run this contract there")
    repo = Path(__file__).parents[1]
    settings = yaml.safe_load((repo / 'configs/inference_v22_selected.yaml').read_text())
    settings['stability'] = gate()
    settings['stability']['enabled'] = enabled
    config_path = tmp_path / 'v23.yaml'
    config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    notices = tmp_path / 'notices'
    notices.mkdir()
    for name in ('LICENSE', 'NOTICE'):
        (notices / name).write_text('Unit test notice; not a submission release')
    package = export_submission(checkpoint, config_path, tmp_path / 'package', notices)
    archive = make_submission_zip(package, tmp_path / 'package.zip')
    import zipfile
    extracted = tmp_path / 'extracted'
    with zipfile.ZipFile(archive) as z:
        assert 'assets/code/terradelta/inference/stability_v23.py' in z.namelist()
        assert not any('/training/' in n or '/data/' in n or n.endswith('.log') for n in z.namelist())
        for name in z.namelist():
            if name.endswith(('.py', '.yaml', '.ipynb', '.txt')):
                assert not re.search(rb'AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----', z.read(name))
        z.extractall(extracted)
    from PIL import Image
    input_dir = tmp_path / 'input'
    folder = input_dir / 'images/0001'
    folder.mkdir(parents=True)
    (input_dir / 'pairs.csv').write_text('id\n0001\n')
    for name in ('pre', 'post'):
        Image.fromarray(np.full((256, 256, 3), 128, np.uint8)).save(folder / (name + '.png'))
    notebook = json.loads((extracted / 'predict.ipynb').read_text())
    code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    prelude = '''import socket,torch
def forbidden(*a,**kw): raise AssertionError('Network/training forbidden')
socket.socket.connect=forbidden
socket.create_connection=forbidden
torch.hub.download_url_to_file=forbidden
torch.optim.Optimizer.__init__=forbidden
torch.Tensor.backward=forbidden
torch.autograd.backward=forbidden
torch.cuda.is_available=lambda:False
'''
    suffix = '''
import sys
for name,module in list(sys.modules.items()):
    if name=='terradelta' or name.startswith('terradelta.'):
        assert Path(module.__file__).resolve().is_relative_to(ROOT/'assets/code'),name
'''
    actual = tmp_path / 'actual.csv'
    run = subprocess.run([os.sys.executable, '-I', '-c', prelude + code + suffix], cwd=extracted,
            env=dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='', AIF_INPUT_DIR=str(input_dir),
                     AIF_PREDICTION_PATH=str(actual)), text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    from terradelta.inference.stability_v23 import predict_directory
    expected = tmp_path / 'expected.csv'
    predict_directory(input_dir, expected, checkpoint, settings)
    assert actual.read_bytes() == expected.read_bytes()
    if not enabled:
        from terradelta.inference.v2 import predict_directory as predict_v22
        reference = tmp_path / 'v22.csv'
        predict_v22(input_dir, reference, checkpoint, settings)
        assert actual.read_bytes() == reference.read_bytes()


def _evaluator():
    path = Path(__file__).parents[1] / 'scripts/evaluate_v23.py'
    spec = importlib.util.spec_from_file_location('evaluate_v23', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strong_promotion_rejects_small_gains_recall_or_shape_loss():
    evaluator = _evaluator()
    baseline = {"num_samples": 22, "no_change_count": 17, "no_change_fp_any": 16, "score": .52269,
                "classes": {name: {"gt_positive_count": count, "presence_confusion": {"tp": count},
                                   "shape_score": .8} for name, count in zip(CLASSES, (3, 2))}}
    good = copy.deepcopy(baseline)
    good["no_change_fp_any"] = 12
    assert evaluator.local_gate(good, baseline)["passed"]
    for fp in (14, 15, 16):
        bad = copy.deepcopy(good)
        bad["no_change_fp_any"] = fp
        assert not evaluator.local_gate(bad, baseline)["passed"]
    for name in CLASSES:
        bad = copy.deepcopy(good)
        bad["classes"][name]["presence_confusion"]["tp"] -= 1
        assert not evaluator.local_gate(bad, baseline)["passed"]
    bad = copy.deepcopy(good)
    bad["score"] = .479
    assert not evaluator.local_gate(bad, baseline)["passed"]
    bad = copy.deepcopy(good)
    bad["classes"][CLASSES[0]]["shape_score"] = .7
    assert not evaluator.local_gate(bad, baseline)["passed"]


def test_stress_selection_uses_soft_plateau_and_never_promotes_recall_loss():
    evaluator = _evaluator()
    baseline = {"num_samples": 22, "no_change_count": 17, "no_change_fp_any": 16, "score": .52269,
                "classes": {name: {"gt_positive_count": count, "presence_confusion": {"tp": count},
                                   "shape_score": .8} for name, count in zip(CLASSES, (3, 2))}}
    regular, plateau, lost = (copy.deepcopy(baseline) for _ in range(3))
    regular.update(no_change_fp_any=13, score=.568)
    plateau.update(no_change_fp_any=13, score=.5675)
    lost.update(no_change_fp_any=8, score=.6)
    lost['classes'][CLASSES[0]]['presence_confusion']['tp'] = 2
    trials = {key: {'combined': value, 'local_gate': evaluator.local_gate(value, baseline)}
              for key, value in [('B', regular), ('class_selected', plateau), ('lost_positive', lost)]}
    assert evaluator.rank_finalists(trials, baseline) == ['class_selected', 'B']


def test_no_change_metrics_match_by_id_and_accept_encoded_empty_truth():
    from terradelta.inference.v2_evaluation import prediction_metrics
    polygon = '[[[0,0],[10,0],[10,10],[0,10]]]'
    predictions = [{'id': 'positive', 'new_building': polygon, 'tree_removal': ''},
                   {'id': 'negative', 'new_building': polygon, 'tree_removal': ''}]
    truth = [{'id': 'negative', 'new_building': '[]', 'tree_removal': ''},
             {'id': 'positive', 'new_building': polygon, 'tree_removal': ''}]
    result = prediction_metrics(predictions, truth)
    assert result['no_change_count'] == result['no_change_fp_any'] == 1
    assert result['classes'][CLASSES[0]]['no_change_fp_count'] == 1
    assert result['classes'][CLASSES[0]]['presence_confusion'] == {'tp': 1, 'fp': 1, 'fn': 0, 'tn': 0}
