"""Safety-focused tests: no test calls fit(), backward(), or optimizer.step()."""
import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from terradelta.training.losses import build_loss
from terradelta.training.splits import assert_training_licenses, prepare_datasets, split_indices
from terradelta.training.trainer import (
    BatchStream, Trainer, accumulation_batches, capture_rng_state, dry_run,
    lr_multiplier, parameter_groups, restore_rng_state, validate_training_config,
)


@pytest.fixture(autouse=True)
def prohibit_optimizer_steps(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer steps are permitted in training tests")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    monkeypatch.setattr(Trainer, "fit", forbidden)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(6, 4, 1), nn.BatchNorm2d(4))
        self.head = nn.Conv2d(4, 3, 1)

    def forward(self, image):
        return self.head(self.encoder(image))


class TinyDataset(Dataset):
    transform = None

    def __init__(self, size=7):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"id": str(index), "image": torch.ones(6, 8, 8) * index,
                "mask": torch.full((8, 8), index % 3, dtype=torch.long)}


def cfg(tmp_path):
    return {"training": {"output_dir": str(tmp_path), "batch_size": 2,
                         "max_optimizer_steps": 100, "loss": {"name": "ce_dice"}}}


@pytest.mark.parametrize("name", ["ce", "weighted_ce", "dice", "ce_dice", "focal", "focal_dice"])
def test_losses_finite_and_ignore_pixels(name):
    logits = torch.tensor([[[[2., -1.]], [[0., 2.]], [[-2., 0.]]]])
    target = torch.tensor([[[0, 1]]])
    loss = build_loss({"name": name, "class_weights": [0.2, 2.0, 1.0]})
    assert loss(logits, target).isfinite()
    ignored = torch.full_like(target, -100)
    assert loss(logits, ignored).item() == 0
    assert loss(torch.cat([logits, torch.zeros_like(logits)], -1), torch.cat([target, ignored], -1)).item() == pytest.approx(loss(logits, target).item())


def test_loss_numerics_and_compositions():
    torch.manual_seed(5)
    logits, target = torch.randn(2, 3, 5, 5), torch.randint(0, 3, (2, 5, 5))
    assert build_loss("ce")(logits, target) == pytest.approx(F.cross_entropy(logits, target).item())
    assert build_loss({"name": "focal", "gamma": 0})(logits, target) == pytest.approx(F.cross_entropy(logits, target).item())
    weights = [0.1, 3., 2.]
    assert build_loss({"name": "weighted_ce", "class_weights": weights})(logits, target) == pytest.approx(F.cross_entropy(logits, target, weight=torch.tensor(weights)).item())
    for base in ("ce", "focal"):
        expected = build_loss(base)(logits, target) + build_loss("dice")(logits, target)
        assert build_loss(f"{base}_dice")(logits, target) == pytest.approx(expected.item())
    assert build_loss({"training": {"batch_size": 2}}).name == "ce_dice"


def test_dice_includes_absent_change_classes_as_false_positives():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    confident_background = torch.full((1, 3, 4, 4), -20.)
    confident_background[:, 0] = 20.
    confident_change = confident_background.flip(1)
    assert build_loss("dice")(confident_background, target) < build_loss("dice")(confident_change, target)


@pytest.mark.parametrize("loss", [{"name": "bad"}, {"name": "weighted_ce"},
                                   {"class_weights": [1, -1, 1]}, {"class_weights": [1, 2]}, {"gamma": float("nan")}])
def test_invalid_losses(loss):
    with pytest.raises(ValueError):
        build_loss(loss)


@pytest.mark.parametrize("strategy,key", [("region", "region_id"), ("state", "state")])
def test_groups_never_leak_and_seed_replays(strategy, key):
    rows = [{"id": str(i), key: f"group_{i // 3}", "region_id": f"region_{i // 3}"} for i in range(18)]
    train, val = split_indices(rows, strategy=strategy, seed=72)
    assert train and val and set(train).isdisjoint(val) and sorted(train + val) == list(range(18))
    assert {rows[i][key] for i in train}.isdisjoint({rows[i][key] for i in val})
    assert (train, val) == split_indices(rows, strategy=strategy, seed=72)


def test_temporal_uses_latest_year_pair_and_random_is_explicit():
    rows = [{"region_id": "a", "year_pre": 2019, "year_post": 2020},
            {"region_id": "b", "year_pre": 2020, "year_post": 2022},
            {"region_id": "c", "year_pre": 2021, "year_post": 2023},
            {"region_id": "c", "year_pre": 2021, "year_post": 2023}]
    train, val = split_indices(rows, "temporal", .3, temporal_year=2023)
    assert val == [2, 3] and train == [0, 1]
    assert split_indices(rows, "random", .3, 4) == split_indices(rows, "random", .3, 4)
    with pytest.raises(ValueError, match="region_id"):
        split_indices([{ "id": "missing-region"}, {"id": "missing-region-2"}])
    with pytest.raises(ValueError, match="two disconnected"):
        split_indices([{"region_id": "a"}, {"region_id": "a"}])


def test_explicit_manifest_region_leak_rejected(tmp_path):
    paths = [tmp_path / "train.csv", tmp_path / "val.csv"]
    for i, path in enumerate(paths):
        path.write_text(f"id,region_id,source,license_status\n{i},same,synthetic,commercial_ok\n")
    config = {"data": {"train_manifest": str(paths[0]), "val_manifest": str(paths[1])}}
    with pytest.raises(ValueError, match="leak region"):
        prepare_datasets(config, dataset_factory=lambda *args, **kwargs: None)


def test_training_provenance_gate_rejects_unknown_and_unreviewed():
    for row in [{"id": "1", "source": "naip", "license_status": "unknown"},
                {"id": "1", "source": "levir_cd", "license_status": "commercial_ok"},
                {"id": "1", "source": "hansen", "license_status": "public_domain"},
                {"id": "1", "source": "naip", "license_status": "commercial_ok", "label_status": "candidate"}]:
        with pytest.raises(ValueError, match="unapproved"):
            assert_training_licenses([row], {})
    assert_training_licenses([{"source": "synthetic", "license_status": "commercial_ok"}], {})


def test_batch_stream_restores_middle_of_epoch_and_next_permutation():
    stream = BatchStream(7, 2, seed=5)
    stream.next_batch()
    stream.next_batch()
    saved = stream.state_dict()
    expected = [stream.next_batch() for _ in range(9)]
    restored = BatchStream(7, 2, seed=5)
    restored.load_state_dict(saved)
    assert [restored.next_batch() for _ in range(9)] == expected
    assert saved["offset"] == 4
    invalid = copy.deepcopy(saved)
    invalid["order"] = [0] * 7
    with pytest.raises(ValueError, match="permutation"):
        restored.load_state_dict(invalid)
    with pytest.raises(ValueError, match="batch_size"):
        BatchStream(7, 3, seed=5).load_state_dict(saved)


def test_partial_accumulation_normalizes_actual_samples_and_stays_in_epoch():
    stream = BatchStream(7, 2, seed=0)
    first = accumulation_batches(stream, 3)
    second = accumulation_batches(stream, 3)
    third = accumulation_batches(stream, 3)
    assert [len(group) for group in (first, second, third)] == [3, 1, 3]
    assert sum(len(item[0]) for item in second) == 1
    assert [item[1] for item in first + second] == [0, 0, 0, 0]
    assert [item[1] for item in third] == [1, 1, 1]
    assert sum(len(indices) / 7 for indices, _, _ in first + second) == pytest.approx(1)


def test_rng_round_trip_safe_serialization(tmp_path):
    state = capture_rng_state()
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    expected = random.random(), np.random.random(), torch.rand(3)
    restore_rng_state(torch.load(path, weights_only=True))
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_encoder_head_groups_do_not_overlap(tmp_path):
    model = TinyModel()
    options = validate_training_config(cfg(tmp_path))
    groups = parameter_groups(model, options)
    encoder, head = [{id(p) for p in group["params"]} for group in groups]
    assert encoder.isdisjoint(head)
    assert encoder | head == {id(p) for p in model.parameters()}
    assert [group["lr"] for group in groups] == [1e-5, 1e-4]


def test_scheduler_math_has_warmup_and_decay_without_steps():
    assert lr_multiplier(0, 100, "cosine", 10, .1) == .1
    assert lr_multiplier(9, 100, "cosine", 10, .1) == 1
    assert lr_multiplier(10, 100, "cosine", 10, .1) == 1
    assert lr_multiplier(100, 100, "cosine", 10, .1) == .1
    assert lr_multiplier(50, 100, "linear") == .5
    assert lr_multiplier(100, 100, "constant") == 1


def test_dry_run_never_constructs_optimizer_and_preserves_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not construct optimizer or scheduler")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", forbidden)
    model = TinyModel().train()
    model.encoder[1].eval()  # Preserve independently frozen module modes too.
    before = {key: value.clone() for key, value in model.state_dict().items()}
    plan = dry_run(model, TinyDataset(), TinyDataset(2), cfg(tmp_path), device="cpu")
    assert plan["optimizer_constructed"] is False and plan["optimizer_steps_executed"] == 0
    assert plan["smoke_logits_shape"] == [1, 3, 8, 8]
    assert model.training and not model.encoder[1].training
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert not list(tmp_path.iterdir())


def test_checkpoint_resume_restores_state_without_training(tmp_path):
    config = cfg(tmp_path)
    model, data = TinyModel(), TinyDataset()
    trainer = Trainer(model, data, TinyDataset(2), config, device="cpu")
    trainer.stream.next_batch()
    trainer.best_metric, trainer.bad_validations = .4, 2
    checkpoint = trainer.save_checkpoint("fixture.pt")
    expected_rng = random.random(), np.random.random(), torch.rand(2)
    expected_batches = [trainer.stream.next_batch() for _ in range(7)]
    resumed_config = copy.deepcopy(config)
    resumed_config["training"]["resume"] = str(checkpoint)
    resumed = Trainer(TinyModel(), data, TinyDataset(2), resumed_config, device="cpu")
    assert resumed.best_metric == .4 and resumed.bad_validations == 2
    assert resumed.global_step == 0
    assert random.random() == expected_rng[0] and np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(2), expected_rng[2])
    assert [resumed.stream.next_batch() for _ in range(7)] == expected_batches
    assert all(torch.equal(value, resumed.model.state_dict()[key]) for key, value in model.state_dict().items())
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    assert resumed.optimizer.state_dict() == trainer.optimizer.state_dict()
    changed = copy.deepcopy(resumed_config)
    changed["training"]["batch_size"] = 3
    with pytest.raises(ValueError, match="configuration/data"):
        Trainer(TinyModel(), data, TinyDataset(2), changed, device="cpu")
    trainer._accumulating = True
    with pytest.raises(RuntimeError, match="partial"):
        trainer.checkpoint_dict()


def test_early_stopping_observation_no_training(tmp_path):
    config = cfg(tmp_path)
    config["training"]["early_stopping"] = {"enabled": True, "metric": "score", "patience": 2, "min_delta": .01}
    trainer = Trainer(TinyModel(), TinyDataset(), TinyDataset(2), config, device="cpu")
    assert trainer._observe({"score": .5}) == (True, False)
    assert trainer._observe({"score": .505}) == (False, False)
    assert trainer._observe({"score": .4}) == (False, True)
    assert trainer.best_metric == .5


@pytest.mark.parametrize("field,value", [("max_optimizer_steps", 0), ("gradient_accumulation_steps", 0),
                                         ("num_workers", 2), ("amp", "true"), ("encoder_lr", float("nan"))])
def test_invalid_training_configuration(field, value):
    with pytest.raises(ValueError):
        validate_training_config({"training": {field: value}})


def test_training_yaml_configs_match_schema():
    from terradelta.utils.io import load_config
    root = Path(__file__).resolve().parents[1]
    short = load_config(root / "configs/train_short.yaml")
    medium = load_config(root / "configs/train_medium.yaml")
    assert validate_training_config(short)["max_optimizer_steps"] == 100
    assert validate_training_config(medium)["max_optimizer_steps"] == 1000
    assert short["data"]["split"]["strategy"] == "region"
    assert short["model"]["encoder_weights"] is None
