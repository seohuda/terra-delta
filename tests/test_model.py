import json
from pathlib import Path

import numpy as np
import pytest
import torch

from terradelta.inference.predictor import prepare_pair, Predictor
from terradelta.models.factory import build_model
from terradelta.models.checkpoint import load_checkpoint
from terradelta.inference.tta import predict_probabilities


def test_model_construction_forward():
    model = build_model().eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 6, 256, 256))
    assert result.shape == (1, 3, 256, 256)
    assert torch.isfinite(result).all()


def test_official_preprocessing_and_forward_parity(official_checkpoint):
    model = build_model().eval()
    load_checkpoint(official_checkpoint, model)
    rng = np.random.default_rng(4)
    pre, post = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
    notebook = json.loads((official_checkpoint.parents[2] / "predict.ipynb").read_text())
    setup = "".join(notebook["cells"][1]["source"])
    model_cell = "".join(notebook["cells"][4]["source"]).split("# 가속기")[0]
    namespace = {}
    exec(setup, namespace)
    exec(model_cell, namespace)
    official_labels = namespace["predict_batch"](model, [pre], [post], "cpu")
    image = prepare_pair(pre, post)
    actual = Predictor(model, {"inference": {"tta": ["identity"]}}).predict_batch(image[None])
    np.testing.assert_array_equal(actual.argmax(1).astype(np.uint8), official_labels)
    assert image.shape == (6, 256, 256)


def test_tta_inverts_probability_orientation():
    class Pointwise(torch.nn.Module):
        def forward(self, x):
            return x[:, :3]
    image = torch.randn(2, 6, 32, 32)
    base = predict_probabilities(Pointwise(), image)
    tta = predict_probabilities(Pointwise(), image, ["identity", "horizontal", "vertical"])
    torch.testing.assert_close(base, tta)
    with pytest.raises(ValueError):
        predict_probabilities(Pointwise(), image, ["identity", "identity"])


def test_factory_rejects_incompatible_architecture():
    with pytest.raises(ValueError):
        build_model({"in_channels": 3})
