"""Frozen CPU cache and resumable polygon calibration. No training dependencies."""
import csv
import hashlib
import json
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from terradelta.metrics import evaluate_predictions
from terradelta.metrics.evaluation import _polygon_union
from terradelta.postprocess import probabilities_to_masks
from terradelta.postprocess.polygons import reference_components, reference_exteriors
from terradelta.training.search import CLASSES, load_ground_truth, load_probability_maps


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def cache_validation(manifest, checkpoint, output):
    """One identity CPU forward per uncached ID; content-bound lossless resume."""
    import torch
    from terradelta.data.dataset import ChangeDataset
    from terradelta.inference.predictor import Predictor
    from terradelta.models.checkpoint import load_checkpoint
    from terradelta.models.factory import build_model

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ChangeDataset(manifest)
    samples = [dataset[i] for i in range(len(dataset))]
    counts = [sum(bool((s['mask'] == k).any()) for s in samples) for k in (1, 2)]
    negatives = sum(not bool((s['mask'] > 0).any()) for s in samples)
    if (len(samples), *counts, negatives) != (22, 3, 2, 17):
        raise ValueError(f'Unexpected validation distribution: {len(samples)}, {counts}, {negatives}')
    if any(not bool(s['valid_mask'].all()) for s in samples):
        raise ValueError('Validation must contain full-tile labels')
    inputs = [{k: digest(row[k]) for k in ('pre', 'post', *CLASSES)} for row in dataset.samples]
    identity = {'checkpoint_sha256': digest(checkpoint), 'manifest_sha256': digest(manifest),
                'inputs': inputs, 'ids': [s['id'] for s in samples], 'dtype': 'float32',
                'tta': ['identity'], 'alignment': 'none', 'normalization': 'ImageNet', 'schema': 1}
    meta = output / 'identity.json'
    if meta.exists() and json.loads(meta.read_text()) != identity:
        raise ValueError('Cache identity changed; use a new output directory')
    if not meta.exists():
        if list(output.glob('*.npz')) or list(output.glob('*.npy')):
            raise ValueError('Unowned probability cache')
        write_json(meta, identity)
    torch.set_num_threads(2)
    missing = [s for s in samples if not (output / f"{s['id']}.npz").exists()]
    if missing:
        model = build_model({'encoder_weights': None})
        load_checkpoint(checkpoint, model)
        predictor = Predictor(model, {'inference': {'tta': ['identity'], 'alignment': 'none'}}, 'cpu')
        for s in missing:
            with torch.inference_mode():
                p = predictor.predict_batch(s['image'][None])[0]
            temp = output / f"{s['id']}.tmp.npz"
            np.savez_compressed(temp, probabilities=p.astype(np.float32))
            temp.replace(output / f"{s['id']}.npz")
            print(f"Cached {s['id']} ({p.dtype}); gradients disabled", flush=True)
    maps = load_probability_maps(output)
    if set(maps) != set(identity['ids']):
        raise ValueError('Unexpected cache IDs')
    for p in maps.values():
        if p.dtype != np.float32 or not np.allclose(p.sum(axis=0), 1, atol=1e-6):
            raise ValueError('Invalid lossless softmax cache')
    if digest(checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    return identity


def preset(building=.5, tree=.5, *, mode='threshold'):
    return {'model': {'encoder': 'resnet18', 'in_channels': 6, 'classes': 3, 'encoder_weights': None},
            'inference': {'tta': ['identity'], 'alignment': 'none', 'batch_size': 1},
            'postprocess': {'mode': mode, 'backend': 'reference', 'ndigits': 2,
                            'morphology': {'opening': 0, 'closing': 0, 'dilation': 0, 'erosion': 0,
                                           'fill_holes': False},
                            'classes': {name: {'threshold': threshold, 'min_area': 30,
                                               'min_pos_area': 20, 'simplify_px': .5}
                                        for name, threshold in zip(CLASSES, (building, tree))}}}


class Calibration:
    """Canonical class metrics with exact reference geometry reuse and trial resume."""
    def __init__(self, probabilities, truth, output):
        self.p = probabilities
        self.truth = truth
        self.ids = [r['id'] for r in truth]
        if len(set(self.ids)) != len(self.ids) or set(self.ids) != set(probabilities):
            raise ValueError('Prediction/GT ID mismatch or duplicate GT')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(json.dumps(truth, sort_keys=True).encode())
        for identifier in sorted(probabilities):
            p = np.asarray(probabilities[identifier])
            h.update(identifier.encode())
            h.update(str((p.shape, p.dtype)).encode())
            h.update(p.tobytes())
        from terradelta.postprocess import thresholds, polygons
        from terradelta.metrics import evaluation
        for module in (thresholds, polygons, evaluation):
            h.update(Path(module.__file__).read_bytes())
        h.update(Path(__file__).read_bytes())
        identity = {'sha256': h.hexdigest(), 'schema': 1}
        path = self.output / 'search_identity.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Search inputs/code changed; cannot resume')
        write_json(path, identity)
        self.trials = {}
        self.journal = self.output / 'trials.jsonl'
        if self.journal.exists():
            # A torn final record is rejected, never silently mistaken for a trial.
            for line in self.journal.read_text().splitlines():
                trial = json.loads(line)
                self.trials[self.key(trial['config'])] = trial
        self.components = {}
        self.class_cache = {}
        self.nochange = [not any(_polygon_union(r[n], n).area > 0 for n in CLASSES) for r in truth]

    @staticmethod
    def key(config):
        return json.dumps(config, sort_keys=True, separators=(',', ':'))

    def rows(self, config):
        rows = [{'id': identifier} for identifier in self.ids]
        for i, identifier in enumerate(self.ids):
            masks = probabilities_to_masks(self.p[identifier], config)
            for name in CLASSES:
                mask = masks[name]
                key = (mask.shape, hashlib.sha256(mask.tobytes()).digest())
                if key not in self.components:
                    self.components[key] = reference_components(mask)
                opts = config['postprocess']['classes'][name]
                export_key = (key, self.key(opts))
                if export_key not in self.class_cache:
                    self.class_cache[export_key] = reference_exteriors(self.components[key], opts['min_area'],
                                                                     opts['min_pos_area'], opts['simplify_px'], 2)
                rows[i][name] = self.class_cache[export_key]
        return rows

    def evaluate(self, config):
        key = self.key(config)
        if key in self.trials:
            return self.trials[key]
        rows = self.rows(config)
        scores = evaluate_predictions(rows, self.truth)
        areas = {n: [_polygon_union(r[n], n).area for r in rows] for n in CLASSES}
        flags = [any(areas[n][i] >= 20 for n in CLASSES) for i in range(len(rows))]
        flat = {'overall': scores['score'], 'no_change_fp_count': sum(f and nc for f, nc in zip(flags, self.nochange)),
                'no_change_count': sum(self.nochange), 'predicted_positive_samples': sum(flags)}
        flat['no_change_fp_rate'] = flat['no_change_fp_count'] / max(1, flat['no_change_count'])
        for n in CLASSES:
            s = scores['classes'][n]
            flat.update({f'{n}.{k}': s[k] for k in ('score', 'presence_macro_f1', 'shape_score')})
            flat.update({f'{n}.{k}': v for k, v in s['presence_confusion'].items()})
            flat[f'{n}.predicted_positive_samples'] = sum(a >= 20 for a in areas[n])
            flat[f'{n}.average_area'] = sum(areas[n]) / len(rows)
            flat[f'{n}.average_polygons'] = sum(len(r[n]) for r in rows) / len(rows)
            flat.update({f'{n}.{k}': v for k, v in config['postprocess']['classes'][n].items()})
        trial = {'config': config, 'metrics': scores, 'summary': flat}
        with self.journal.open('a') as f:
            f.write(json.dumps(trial, sort_keys=True, allow_nan=False) + '\n')
            f.flush()
        self.trials[key] = trial
        return trial

    def stage(self, bases, parameter, values, filename, *, shared=False):
        trials = {}
        combinations = [(v, v) for v in values] if shared else list(product(values, repeat=2))
        for base in bases:
            for pair in combinations:
                c = deepcopy(base)
                for n, v in zip(CLASSES, pair):
                    c['postprocess']['classes'][n][parameter] = v
                t = self.evaluate(c)
                trials[self.key(c)] = t
            print(f'{filename}: {len(trials)} candidates complete', flush=True)
        self.csv(filename, list(trials.values()))
        return list(trials.values())

    def csv(self, filename, trials):
        path = self.output / filename
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(trials[0]['summary']))
            writer.writeheader()
            writer.writerows(t['summary'] for t in trials)


def rank(trial, kind='balanced'):
    s = trial['summary']
    fn = sum(s[f'{n}.fn'] for n in CLASSES)
    fp, score = s['no_change_fp_rate'], s['overall']
    settings = trial['config']['postprocess']['classes']
    # Equal metrics retain original numerical defaults rather than selecting
    # the first redundant total-area/simplification value in the grid.
    defaults = -sum(abs(settings[n][p] - v) for n in CLASSES
                    for p, v in (('min_area', 30), ('min_pos_area', 20), ('simplify_px', .5)))
    # Presence recall is first for all presets. Balanced targets <=30% before
    # score; conservative minimizes FP; aggressive optimizes score then shape.
    if kind == 'conservative':
        return (-fn, -fp, score, defaults)
    if kind == 'aggressive':
        return (-fn, score, sum(s[f'{n}.shape_score'] for n in CLASSES), -fp, defaults)
    return (-fn, fp <= .30, score, -fp, defaults)


def finalists(trials):
    selected = []
    for kind in ('conservative', 'balanced', 'aggressive'):
        t = max(trials, key=lambda t: rank(t, kind))
        if t['config'] not in selected:
            selected.append(t['config'])
    return selected


def run_sweep(pred_dir, manifest, output, baseline_score):
    engine = Calibration(load_probability_maps(pred_dir), load_ground_truth(manifest), output)
    baseline = engine.evaluate(preset(mode='argmax'))
    if abs(baseline['summary']['overall'] - baseline_score) > 1e-8:
        raise ValueError(f"Baseline mismatch: {baseline['summary']['overall']} != {baseline_score}")
    # Assert the cached component adapter matches the original converter.
    from terradelta.training.validation import prediction_row
    canonical = [prediction_row(i, engine.p[i], baseline['config']) for i in engine.ids]
    if evaluate_predictions(canonical, engine.truth) != baseline['metrics']:
        raise ValueError('Reference component cache/evaluator parity failed')
    print(f"Reproduced baseline {baseline['summary']['overall']:.12f}", flush=True)
    coarse = engine.stage([preset()], 'threshold', [round(.30 + .05*i, 2) for i in range(13)], 'coarse_thresholds.csv')
    # Retain three distinct ranked threshold regions, including recall/FP aims.
    bases = finalists(coarse)
    for t in sorted(coarse, key=rank, reverse=True):
        if len(bases) >= 3:
            break
        if t['config'] not in bases:
            bases.append(t['config'])
    area = engine.stage(bases, 'min_area', [10, 20, 30, 40, 50, 75, 100, 150, 200], 'area_search.csv')
    total = engine.stage(finalists(area), 'min_pos_area', [10, 20, 30, 50, 75, 100, 150], 'positive_area_search.csv')
    final = engine.stage(finalists(total), 'simplify_px', [0, .25, .5, .75, 1, 1.5], 'final_search.csv', shared=True)
    chosen = {kind: max(final, key=lambda t: rank(t, kind)) for kind in ('conservative', 'balanced', 'aggressive')}
    # Require a recall-preserving stable plateau before comparing centers.
    # Prefer centers that improve the baseline, then lower FP. A tiny gain in
    # neighbor mean must not select a center with substantially more FP.
    neighborhoods = []
    centers = {}
    for finalist in finalists(final):
        b, t = [finalist['postprocess']['classes'][n]['threshold'] for n in CLASSES]
        for db, dt in product((-.05, 0, .05), repeat=2):
            pair = (round(b+db, 2), round(t+dt, 2))
            if not all(.30 <= v <= .90 for v in pair):
                continue
            c = deepcopy(finalist)
            for n, v in zip(CLASSES, pair):
                c['postprocess']['classes'][n]['threshold'] = v
            centers[engine.key(c)] = c
    for c in centers.values():
        neighbor = []
        b, t = [c['postprocess']['classes'][n]['threshold'] for n in CLASSES]
        for db, dt in product((-.05, 0, .05), repeat=2):
            d = deepcopy(c)
            for n, v in zip(CLASSES, (b+db, t+dt)):
                d['postprocess']['classes'][n]['threshold'] = round(min(1, max(0, v)), 2)
            neighbor.append(engine.evaluate(d))
        worst_fn = max(sum(x['summary'][f'{n}.fn'] for n in CLASSES) for x in neighbor)
        worst_fp = max(x['summary']['no_change_fp_rate'] for x in neighbor)
        mean = sum(x['summary']['overall'] for x in neighbor) / len(neighbor)
        spread = max(x['summary']['overall'] for x in neighbor) - min(x['summary']['overall'] for x in neighbor)
        center = engine.evaluate(c)
        s = center['summary']
        priority = (-worst_fn, spread <= .05, s['overall'] >= baseline['summary']['overall'],
                    -s['no_change_fp_rate'], s['overall'], -worst_fp, mean)
        neighborhoods.append((priority, center, neighbor))
    _, balanced, neighbors = max(neighborhoods, key=lambda x: x[0])
    chosen['balanced'] = balanced
    engine.csv('robustness.csv', neighbors)
    engine.csv('plateau_centers.csv', [item[1] for item in neighborhoods])
    result = {'baseline': baseline, 'presets': chosen, 'robustness': [t['summary'] for t in neighbors],
              'trial_count': len(engine.trials), 'morphology_searched': False,
              'selection': 'recall first; conservative lowest FP; balanced stable recall plateau, nonregressing center, lower central FP then score; aggressive central score',
              'metric_label': 'approximate local metric; AI-reviewed tiny validation, not a leaderboard estimate'}
    write_json(Path(output) / 'best_configs.json', result)
    for kind, trial in chosen.items():
        (Path(output) / f'inference_sweep_{kind}.yaml').write_text(yaml.safe_dump(trial['config'], sort_keys=False))
    return engine, result


def analysis_artifacts(engine, result, manifest):
    """Per-sample presence transitions and bounded native comparison previews."""
    from PIL import Image, ImageDraw
    from terradelta.data.dataset import read_manifest

    baseline = engine.rows(result['baseline']['config'])
    balanced = engine.rows(result['presets']['balanced']['config'])
    conservative = engine.rows(result['presets']['conservative']['config'])
    changes = []
    coarse_path = engine.output / 'coarse_thresholds.csv'
    coarse = list(csv.DictReader(coarse_path.open()))
    coarse_configs = [t['config'] for t in engine.trials.values()
                      if t['config']['postprocess']['mode'] == 'threshold'
                      and all(t['config']['postprocess']['classes'][n]['min_area'] == 30
                              and t['config']['postprocess']['classes'][n]['min_pos_area'] == 20
                              and t['config']['postprocess']['classes'][n]['simplify_px'] == .5 for n in CLASSES)]
    coarse_keys = {(float(r['new_building.threshold']), float(r['tree_removal.threshold'])) for r in coarse}
    thresholds_fixed = {i: [] for i in engine.ids}
    for c in coarse_configs:
        pair = tuple(c['postprocess']['classes'][n]['threshold'] for n in CLASSES)
        if pair not in coarse_keys:
            continue
        for r in engine.rows(c):
            if _polygon_union(r['new_building'], 'building').area < 20:
                thresholds_fixed[r['id']].append(pair)
    totals = {}
    for n in CLASSES:
        counts = {k: 0 for k in ('FP_fixed', 'FP_still', 'TP_preserved', 'TP_lost', 'new_FP', 'new_FN')}
        for i, identifier in enumerate(engine.ids):
            gt = _polygon_union(engine.truth[i][n], n).area > 0
            old = _polygon_union(baseline[i][n], n).area >= 20
            new = _polygon_union(balanced[i][n], n).area >= 20
            transition = ('TP_preserved' if new else 'TP_lost') if gt and old else (
                ('FP_still' if new else 'FP_fixed') if not gt and old else
                ('new_FN' if gt and not new else 'new_FP' if not gt and new else 'unchanged'))
            if transition in counts:
                counts[transition] += 1
            changes.append({'id': identifier, 'class': n, 'gt_positive': gt,
                            'baseline_positive': old, 'balanced_positive': new, 'transition': transition,
                            'newly_introduced_fn': gt and old and not new,
                            'baseline_area': _polygon_union(baseline[i][n], n).area,
                            'balanced_area': _polygon_union(balanced[i][n], n).area,
                            'coarse_absent_threshold_pairs': json.dumps(sorted(set(thresholds_fixed[identifier])))
                            if n == 'new_building' and old and not gt else ''})
        totals[n] = counts
        totals[n]['newly_introduced_fn'] = counts['TP_lost']
    with (engine.output / 'per_sample_changes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    result['transitions'] = totals
    positive_ids = [i for i, nc in zip(engine.ids, engine.nochange) if not nc]
    hard_ids = list(dict.fromkeys(r['id'] for r in changes if r['transition'] in ('FP_still', 'FP_fixed')))
    selected = list(dict.fromkeys(positive_ids + hard_ids[:5]))
    source = {r['id']: r for r in read_manifest(manifest)}
    board = Image.new('RGB', (256 * 6, 280 * len(selected)), '#101820')
    draw = ImageDraw.Draw(board)
    colors = {'new_building': (255, 80, 70, 95), 'tree_removal': (45, 205, 255, 95)}
    for row, identifier in enumerate(selected):
        i = engine.ids.index(identifier)
        pre = Image.open(source[identifier]['pre']).convert('RGB')
        post = Image.open(source[identifier]['post']).convert('RGB')
        images = [pre, post]
        for polygons in (engine.truth[i], baseline[i], balanced[i], conservative[i]):
            tile = post.convert('RGBA')
            overlay = Image.new('RGBA', tile.size)
            pen = ImageDraw.Draw(overlay)
            for n in CLASSES:
                value = polygons[n]
                if isinstance(value, str):
                    value = json.loads(value) if value else []
                for exterior in value:
                    pen.polygon([tuple(p) for p in exterior], fill=colors[n], outline=colors[n][:3]+(255,))
            images.append(Image.alpha_composite(tile, overlay).convert('RGB'))
        for col, (label, tile) in enumerate(zip(('PRE', 'POST', 'GT', 'baseline', 'balanced', 'conservative'), images)):
            board.paste(tile, (col*256, row*280+24))
            draw.text((col*256+3, row*280+3), f'{identifier} {label}', fill='white')
    previews = engine.output / 'previews'
    previews.mkdir(exist_ok=True)
    board.save(previews / 'comparison.jpg', quality=80, optimize=True)
    result['preview_ids'] = selected
    neighbor = result['robustness']
    result['robustness_summary'] = {
        'minimum_score': min(r['overall'] for r in neighbor), 'maximum_score': max(r['overall'] for r in neighbor),
        'maximum_no_change_fp_rate': max(r['no_change_fp_rate'] for r in neighbor),
        'maximum_fn': max(sum(r[f'{n}.fn'] for n in CLASSES) for r in neighbor)}
    result['robustness_summary']['unstable'] = (result['robustness_summary']['maximum_fn'] > 0 or
        result['robustness_summary']['maximum_score'] - result['robustness_summary']['minimum_score'] > .05)
    write_json(engine.output / 'best_configs.json', result)
