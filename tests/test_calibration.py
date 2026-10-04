import json
import numpy as np
import pytest
import yaml

from terradelta.inference.calibration import Calibration, preset
from terradelta.postprocess.polygons import mask_to_polygons_reference, reference_components, reference_exteriors
from terradelta.training.search import load_probability_maps


def test_reference_component_reuse_exact_parity():
    rng = np.random.default_rng(8)
    mask = rng.random((30, 30)) > .45
    parts = reference_components(mask)
    for area in (0, 10, 30):
        for total in (0, 50, 100):
            for simplify in (0, .5, 1.5):
                expected = mask_to_polygons_reference(mask, area, total, simplify)
                assert reference_exteriors(parts, area, total, simplify) == (json.loads(expected) if expected else [])


def test_npz_loader_and_duplicate_ids(tmp_path):
    p = np.zeros((3, 4, 4), np.float32)
    p[0] = 1
    np.savez_compressed(tmp_path / '001.npz', probabilities=p)
    assert np.array_equal(load_probability_maps(tmp_path)['001'], p)
    np.save(tmp_path / '001.npy', p)
    with pytest.raises(ValueError, match='Duplicate'):
        load_probability_maps(tmp_path)


def test_deterministic_resume_fingerprint_and_export(tmp_path):
    p = np.zeros((3, 10, 10), np.float32)
    p[1] = .8
    p[0] = .2
    truth = [{'id': 'a', 'new_building': [[[0, 0], [10, 0], [10, 10], [0, 10]]], 'tree_removal': []}]
    e = Calibration({'a': p}, truth, tmp_path)
    trials = e.stage([preset()], 'threshold', [.5, .9], 'coarse.csv')
    journal = e.journal.read_bytes()
    resumed = Calibration({'a': p}, truth, tmp_path)
    assert resumed.stage([preset()], 'threshold', [.5, .9], 'coarse.csv') == trials
    assert resumed.journal.read_bytes() == journal
    config = yaml.safe_load(yaml.safe_dump(trials[0]['config']))
    assert config == trials[0]['config']
    p[0] = .1
    with pytest.raises(ValueError, match='changed'):
        Calibration({'a': p}, truth, tmp_path)
