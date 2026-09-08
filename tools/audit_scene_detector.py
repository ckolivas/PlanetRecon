"""Checkpointed full optical model consistency and correlated stack-bias study."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.audit_full_gate1 import file_hash, write_json
from tools.experiment_stages import StageStore
from tools.input_compatibility import archived_input_errors, BRIDGE


def stack_statistics(residuals, variances):
    residuals, variances = np.asarray(residuals), np.asarray(variances)
    if residuals.shape != variances.shape or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError('matching positive finite variances required')
    return {'per_frame_mean_standardized_square': float(np.mean(residuals**2/variances)),
            'stack_mean_standardized_square': float(np.mean(residuals.sum(axis=0)**2/variances.sum(axis=0))),
            'stack_max_standardized_square': float(np.max(residuals.sum(axis=0)**2/variances.sum(axis=0)))}


def case(payload):
    import h5py
    from planetrecon.config import SimConfig
    from planetrecon.atmosphere import generate_screen, finite_exposure_psf
    from planetrecon.optics import make_pupil
    from planetrecon.rng import screen_rng
    from planetrecon.operators import SceneDetectorOperator
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(2)
    path, directory, protocol, deadline = payload
    if errors := archived_input_errors(path):
        raise ValueError(f'{path}: {errors}')
    identity = file_hash(path)
    with h5py.File(path) as f:
        cfg_raw = json.loads(f['config/json_utf8'][()])
        cfg = SimConfig(**{k.name: cfg_raw[k.name] for k in fields(SimConfig)})
        # Stored truth is used only to test the operator; never to fill a solve's margins.
        scene = f['object/full_latent_4x'][...].astype(float)
        from planetrecon import constants as C
        flux = float(f['config'].attrs['source_rate_scale'])*cfg.texp_s/C.T0_S
        origins = [tuple(f[f'object/{c}_crop_origin'][...]) for c in ('feature', 'bland')]
        truth = [f[f'object/{c}_truth'][...].astype(float)*flux for c in ('feature', 'bland')]
        expected = [f[f'frames/{c}_expected_e'][...].astype(float) for c in ('feature', 'bland')]
        otfs = f['truth_transfer/feature/otf_bar'][...]
    pupil = make_pupil(cfg)
    screen = generate_screen(cfg, screen_rng(cfg.seed))
    indices = np.unique(np.linspace(0, cfg.n_frames-1, protocol['frames'], dtype=int))
    store = StageStore(directory/f'stages-{cfg.seed}-{int(cfg.dr0)}',
                       {'protocol': protocol, 'input_sha256': identity}, deadline=deadline)
    records = []
    for index in indices:
        def compute():
            psf = finite_exposure_psf(pupil, screen, index*cfg.dt_s, cfg.texp_s,
                                     cfg.exposure_samples_j, cfg.wind_m_s)
            op = SceneDetectorOperator(scene.shape, (psf,), cfg.bin_factor, (0, 0),
                                       (scene.shape[0]//cfg.bin_factor, scene.shape[1]//cfg.bin_factor), flux=flux)
            full = op.forward(scene)
            residuals, variance = [], []
            for j, (ox, oy) in enumerate(origins):
                exp = expected[j][index]
                corrected = full[oy:oy+cfg.eval_size, ox:ox+cfg.eval_size]
                legacy = np.fft.ifft2(otfs[index]*np.fft.fft2(truth[j])).real
                residuals.append(np.stack((corrected-exp, legacy-exp)))
                variance.append(np.maximum(exp, 0.)+C.READ_NOISE_E**2)
            return np.stack(residuals), {'variance': np.stack(variance)}
        records.append(store.run(f'frame-{index:04d}', compute))
        if (len(records) % 50 == 0) or len(records) == len(indices):
            print(f'{cfg.seed} D/r0={cfg.dr0}: {len(records)}/{len(indices)} frames', flush=True)
    rows = []
    for j, crop in enumerate(('feature', 'bland')):
        variance = np.stack([r[1]['variance'][j] for r in records])
        models = {name: stack_statistics(np.stack([r[0][j, m] for r in records]), variance)
                  for m, name in enumerate(('extended', 'legacy_circular'))}
        passed = (models['extended']['per_frame_mean_standardized_square'] <= protocol['per_frame_tolerance']
                  and models['extended']['stack_max_standardized_square'] <= protocol['stack_max_tolerance'])
        rows.append({'crop': crop, 'models': models, 'numerical_consistency_passed': passed})
    row = {'seed': cfg.seed, 'dr0': cfg.dr0, 'indices': indices.tolist(), 'crops': rows,
           'input_sha256': identity, 'input_unchanged': identity == file_hash(path)}
    write_json(directory/f'case-{cfg.seed}-{int(cfg.dr0)}.json', row)
    return row


def run(inputs, directory, *, frames=10, workers=3, resume=False, wall_budget_s=1800.):
    from planetrecon.provenance import source_hash
    from planetrecon.evaluate import family_paths
    from planetrecon import constants as C
    if not 1 <= frames <= 500 or not 1 <= workers <= 16 or not np.isfinite(wall_budget_s) or wall_budget_s <= 0:
        raise ValueError('invalid frames, workers or wall budget')
    directory.mkdir(parents=True, exist_ok=resume)
    protocol = {'question': 'Does extended optical image formation agree with stored means, including correlated stack bias?',
                'source_hash': source_hash(), 'runner_sha256': file_hash(__file__),
                'bridge_sha256': file_hash(BRIDGE), 'checkpoint_sha256': file_hash(Path(__file__).with_name('experiment_stages.py')),
                'compatibility_sha256': file_hash(Path(__file__).with_name('input_compatibility.py')),
                'input_sha256': {p.name: file_hash(p) for p in family_paths(inputs, C.DEV_SEEDS)},
                'frames': frames, 'frame_selection': 'uniformly spaced including first and last',
                'workers': workers, 'threads_per_worker': 2, 'wall_budget_s': wall_budget_s,
                'per_frame_tolerance': 1e-8, 'stack_max_tolerance': 1e-6,
                'decision': 'Numerical consistency only, including stored float32 scene/mean rounding. No reconstruction/physical-model adequacy claim.',
                'q3_authorized': False}
    path = directory/'protocol.json'
    if resume:
        if json.loads(path.read_text()) != protocol:
            raise ValueError('resume identity mismatch')
    else:
        write_json(path, protocol)
    started = time.monotonic()
    cases, failures = [], []
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(case, (p, directory, protocol, started+wall_budget_s)): p
                   for p in family_paths(inputs, C.DEV_SEEDS)}
        for future in as_completed(pending):
            try:
                cases.append(future.result())
            except Exception as exc:
                failures.append({'input': pending[future].name, 'error': repr(exc)})
            write_json(directory/'progress.json', {'cases': cases, 'failures': failures})
    unchanged = (protocol['source_hash'] == source_hash() and protocol['runner_sha256'] == file_hash(__file__)
                 and protocol['checkpoint_sha256'] == file_hash(Path(__file__).with_name('experiment_stages.py'))
                 and protocol['compatibility_sha256'] == file_hash(Path(__file__).with_name('input_compatibility.py'))
                 and protocol['bridge_sha256'] == file_hash(BRIDGE))
    complete = len(cases) == 6 and not failures
    passed = complete and unchanged and all(r['input_unchanged'] and all(c['numerical_consistency_passed'] for c in r['crops']) for r in cases)
    report = {'status': 'valid' if passed else 'incomplete', 'scope': protocol['decision'],
              'q3_authorized': False, 'complete': complete, 'source_unchanged': unchanged,
              'numerical_consistency_passed': passed, 'cases': sorted(cases, key=lambda r: (r['seed'], r['dr0'])),
              'failures': failures, 'wall_s': time.monotonic()-started}
    write_json(directory/f'attempt-{time.time_ns()}.json', report)
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--wall-budget-s', type=float, default=1800.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    report = run(args.inputs, args.out, frames=args.frames, workers=args.workers, resume=args.resume, wall_budget_s=args.wall_budget_s)
    sys.exit(0 if report['numerical_consistency_passed'] else 1)
