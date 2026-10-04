import copy

import numpy as np
import pytest
import torch

from terradelta.inference.v2 import V2Predictor, output_row
from terradelta.inference.verifier import (
    FEATURES, FROZEN_V2_SHA256, change_features, validate_verifier, verifier_scores, verify_frozen_checkpoint,
)
from terradelta.training.verifier import fit_verifier


def config():
    return {"version": 1, "features": list(FEATURES), "checkpoint_sha256": FROZEN_V2_SHA256,
            "mean": np.zeros((2, len(FEATURES))).tolist(), "scale": np.ones((2, len(FEATURES))).tolist(),
            "weight": np.zeros((2, len(FEATURES))).tolist(), "bias": [-3., 3.],
            "threshold": [.5, .5], "pixel_thresholds": [.35, .7]}


def test_identity_and_real_change_features_are_finite():
    rng = np.random.default_rng(4)
    pre = rng.uniform(0, 1, (3, 256, 256))
    post = pre.copy()
    post[:, 50:100, 50:100] = .9
    mean, std = np.array([.485, .456, .406])[:, None, None], np.array([.229, .224, .225])[:, None, None]
    images = np.stack([np.concatenate([(pre - mean) / std, (x - mean) / std]) for x in (pre, post)])
    pixels = np.zeros((2, 2, 256, 256))
    pixels[:, :, 50:100, 50:100] = .9
    features = change_features(images, pixels, np.ones((2, 2)) * .8, [.35, .7])
    assert features.shape == (2, 2, len(FEATURES)) and np.isfinite(features).all()
    assert np.max(np.abs(features[0, :, FEATURES.index("roi_rgb_difference")])) == 0
    assert (features[1, :, FEATURES.index("roi_corrected_difference")] > .1).all()
    empty = change_features(images[:1], np.zeros_like(pixels[:1]), np.zeros((1, 2)), [.35, .7])
    assert np.isfinite(empty).all()


@pytest.mark.parametrize("key,value", [("scale", [[0.] * len(FEATURES)] * 2), ("weight", [[float('nan')] * len(FEATURES)] * 2),
                                      ("threshold", [-1., 1.]), ("pixel_thresholds", [1.1, .5]),
                                      ("checkpoint_sha256", "other"), ("features", list(reversed(FEATURES)))])
def test_bad_verifier_rejected(key, value):
    bad = config()
    bad[key] = value
    with pytest.raises(ValueError):
        validate_verifier(bad)


def test_wrong_frozen_checkpoint_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"wrong frozen checkpoint")
    with pytest.raises(ValueError, match="exact frozen"):
        verify_frozen_checkpoint(path)


def test_outside_training_support_keeps_only_that_class():
    settings = config()
    settings.update(version=2, support_lower=np.full((2, len(FEATURES)), -1).tolist(),
                    support_upper=np.ones((2, len(FEATURES))).tolist(), bias=[-3., -3.])
    features = np.zeros((1, 2, len(FEATURES)))
    assert (verifier_scores(features, settings) < .5).all()
    features[0, 0, 0] = 2.
    scores = verifier_scores(features, settings)
    assert scores[0, 0] == 1. and scores[0, 1] < .5
    settings["support_upper"][0][0] = -2
    with pytest.raises(ValueError, match="reversed"):
        validate_verifier(settings)


def test_only_rejected_class_changes_and_model_state_is_frozen():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(4.))
        def forward(self, image):
            return {"segmentation": self.logit.expand(len(image), 2, 256, 256),
                    "presence": self.logit.expand(len(image), 2)}
    model = Model().requires_grad_(False)
    state = copy.deepcopy(model.state_dict())
    settings = {"verifier": config(), "postprocess": {"presence_threshold": None,
                "classes": {"new_building": {"pixel_threshold": .35}, "tree_removal": {"pixel_threshold": .7}}}}
    predictor = V2Predictor(None, settings, model=model)
    images = torch.zeros(1, 6, 256, 256)
    pixels, presence = predictor.probabilities(images)
    row = predictor.output_rows(["001"], images, pixels, presence)[0]
    reference = output_row("001", pixels[0], presence[0], {})
    assert row["id"] == "001" and row["new_building"] == ""
    assert row["tree_removal"] == reference["tree_removal"]
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match="requires decisions"):
        output_row("001", pixels[0], presence[0], settings)
    with pytest.raises(ValueError, match="batch lengths"):
        predictor.output_rows([], images, pixels, presence)


def test_fit_ignores_partial_absence_and_separates_classes():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(100, 2, len(FEATURES)))
    labels = (features[:, :, 0] > 0).astype(float)
    valid = np.ones((100, 2), bool)
    valid[0] = False
    real = np.arange(100) % 3 == 0
    first = fit_verifier(features, labels, valid, real, [.35, .7])
    labels[0] = 1 - labels[0]
    features[0] = 1e9
    second = fit_verifier(features, labels, valid, real, [.35, .7])
    assert first == second
    features[0] = 0
    scores = verifier_scores(features, first)
    assert np.mean((scores[1:] >= .5) == labels[1:]) > .95
