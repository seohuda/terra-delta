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


def test_per_sample_artifacts_and_bounded_six_column_preview(tmp_path):
    import csv
    from PIL import Image
    from terradelta.inference.calibration import analysis_artifacts

    image = tmp_path / 'image.png'
    Image.new('RGB', (256, 256), 'gray').save(image)
    manifest = tmp_path / 'val.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'pre', 'post'])
        writer.writeheader()
        writer.writerow({'id': 'a', 'pre': image, 'post': image})
    p = np.zeros((3, 256, 256), np.float32)
    p[0] = 1
    p[:, 2:12, 2:12] = np.array([.2, .8, 0])[:, None, None]
    truth = [{'id': 'a', 'new_building': [[[2, 2], [12, 2], [12, 12], [2, 12]]], 'tree_removal': []}]
    e = Calibration({'a': p}, truth, tmp_path / 'results')
    t = e.stage([preset()], 'threshold', [.5], 'coarse_thresholds.csv')[0]
    result = {'baseline': e.evaluate(preset(mode='argmax')),
              'presets': {'balanced': t, 'conservative': t}, 'robustness': [t['summary']]}
    analysis_artifacts(e, result, manifest)
    assert result['transitions']['new_building']['TP_preserved'] == 1
    assert result['transitions']['new_building']['newly_introduced_fn'] == 0
    assert result['robustness_summary']['unstable'] is False
    preview = e.output / 'previews/comparison.jpg'
    assert Image.open(preview).size == (1536, 280)
    assert preview.stat().st_size < 100_000


def test_complete_sweep_selects_recall_plateau_and_resumes(tmp_path):
    import csv
    from terradelta.inference.calibration import run_sweep
    from terradelta.postprocess.polygons import serialize_polygons

    maps = tmp_path / 'maps'
    maps.mkdir()
    truth = []
    for identifier, channel in (('building', 1), ('tree', 2), ('negative', 0)):
        p = np.zeros((3, 10, 10), np.float32)
        p[0] = 1
        if channel == 1:
            p[:, 1:7, 1:7] = np.array([.2, .7, .1])[:, None, None]
        if channel == 2:
            p[:, 1:7, 1:7] = np.array([.34, .30, .36])[:, None, None]
        np.savez_compressed(maps / f'{identifier}.npz', probabilities=p)
        truth.append({'id': identifier, 'new_building': '', 'tree_removal': ''})
        if channel:
            truth[-1][('new_building', 'tree_removal')[channel-1]] = serialize_polygons(
                [[[1, 1], [7, 1], [7, 7], [1, 7]]])
    gt = tmp_path / 'gt.csv'
    with gt.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    engine, result = run_sweep(maps, gt, tmp_path / 'out', 1)
    assert len(result['robustness']) == 9
    assert all(s['tree_removal.fn'] == 0 and s['new_building.fn'] == 0 for s in result['robustness'])
    assert result['presets']['balanced']['config']['postprocess']['classes']['tree_removal']['threshold'] == .3
    before = engine.journal.read_bytes()
    _, repeated = run_sweep(maps, gt, tmp_path / 'out', 1)
    assert repeated == result
    assert engine.journal.read_bytes() == before
    assert len(list((tmp_path / 'out').glob('inference_sweep_*.yaml'))) == 3
