"""Bounded full-optical-grid known-transfer solver pilot; no Gate-1/Q2 claim."""
import argparse
from dataclasses import fields
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from tools.audit_full_gate1 import file_hash, write_json
from tools.experiment_stages import StageStore
from tools.input_compatibility import archived_input_errors, BRIDGE
from tools.scene_quadratic import SceneQuadratic, solve


def run(path, directory, *, crop='feature', frames=3, budgets=(100, 200), ridge=.001,
        noise='observed', weighting='scalar', resume=False, wall_budget_s=300.):
    from planetrecon import constants as C
    from planetrecon.provenance import source_hash
    from planetrecon.config import SimConfig
    from planetrecon.atmosphere import generate_screen, finite_exposure_psf
    from planetrecon.optics import make_pupil
    from planetrecon.rng import screen_rng
    from planetrecon.operators import SceneDetectorOperator
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    if errors := archived_input_errors(path):
        raise ValueError(errors)
    if not 1 <= frames <= 500 or len(budgets) != 2 or budgets[0] < 1 or budgets[1] <= budgets[0]:
        raise ValueError('invalid frame or solver budgets')
    if crop not in ('feature', 'bland') or noise not in ('observed', 'expected') or weighting not in ('scalar', 'spatial'):
        raise ValueError('invalid crop/noise/weighting')
    if ridge <= 0 or not np.isfinite(ridge) or wall_budget_s <= 0 or not np.isfinite(wall_budget_s):
        raise ValueError('positive ridge and wall budget required')
    directory.mkdir(parents=True, exist_ok=resume)
    protocol = {'question': 'Can a known-transfer full-optical-domain pilot certify its regularized solution and doubled-budget stability?',
                'source_hash': source_hash(), 'runner_sha256': file_hash(__file__),
                'solver_sha256': file_hash(Path(__file__).with_name('scene_quadratic.py')),
                'checkpoint_sha256': file_hash(Path(__file__).with_name('experiment_stages.py')),
                'compatibility_sha256': file_hash(Path(__file__).with_name('input_compatibility.py')),
                'bridge_sha256': file_hash(BRIDGE), 'input_sha256': file_hash(path),
                'crop': crop, 'frames': frames, 'selection': 'uniform including sequence endpoints',
                'budgets': list(budgets), 'ridge': ridge, 'smoothness': 0., 'initialization': 'zero',
                'noise': noise, 'weighting': weighting, 'variance_source': 'oracle expected+read diagnostic',
                'unknowns': 'all optical cells, including margins; units=electrons per optical cell',
                'boundary': 'zero exterior beyond original extended scene, positive ridge over all unknowns',
                'threads': 2, 'wall_budget_s': wall_budget_s, 'tolerance': 1e-5,
                'image_stability_tolerance': 1e-4, 'q3_authorized': False,
                'decision': 'Numerical pilot only; prior/domain/noise sensitivity and complete development selections remain unqualified.'}
    dest = directory/'protocol.json'
    if resume:
        if json.loads(dest.read_text()) != protocol:
            raise ValueError('resume identity mismatch')
    else:
        write_json(dest, protocol)
    started = time.monotonic()
    with h5py.File(path) as f:
        raw = json.loads(f['config/json_utf8'][()])
        cfg = SimConfig(**{k.name: raw[k.name] for k in fields(SimConfig)})
        indices = np.unique(np.linspace(0, cfg.n_frames-1, frames, dtype=int))
        # Inspect only the latent dataset shape, never its values.
        shape = f['object/full_latent_4x'].shape
        origin = tuple(f[f'object/{crop}_crop_origin'][...])
        expected = f[f'frames/{crop}_expected_e'][indices].astype(float)
        images = f[f'frames/{crop}_{noise}_e'][indices].astype(float)
    pupil, screen = make_pupil(cfg), generate_screen(cfg, screen_rng(cfg.seed))
    operators = []
    for index in indices:
        psf = finite_exposure_psf(pupil, screen, index*cfg.dt_s, cfg.texp_s,
                                 cfg.exposure_samples_j, cfg.wind_m_s)
        operators.append(SceneDetectorOperator(shape, (psf,), cfg.bin_factor, origin,
                                               (cfg.eval_size, cfg.eval_size)))
    variances = np.maximum(expected, 0.)+C.READ_NOISE_E**2
    if weighting == 'scalar':
        variances = list(variances.mean(axis=(1, 2)))
    problem = SceneQuadratic(operators, images, variances, ridge=ridge)
    store = StageStore(directory/'stages', protocol, deadline=started+wall_budget_s)
    rows, reconstructions = [], []
    for budget in budgets:
        trace = []
        def callback(n, x, certificate):
            trace.append({'iteration': n, **certificate})
            write_json(directory/f'progress-{budget}.json', {'trace': trace})
            print(f'{noise}/{weighting} budget={budget} iteration={n} bound={certificate["relative_solution_error_bound"]:.3g}', flush=True)
        def compute():
            return solve(problem, maxiter=budget, tolerance=protocol['tolerance'],
                         callback=callback, deadline=started+wall_budget_s)
        try:
            image, info = store.run(f'budget-{budget}', compute)
        except TimeoutError as exc:
            rows.append({'maxiter': budget, 'status': 'incomplete', 'reason': str(exc), 'converged': False})
            break
        rows.append(info)
        reconstructions.append(image)
    change = None if len(reconstructions) < 2 else float(np.linalg.norm(reconstructions[1]-reconstructions[0])/max(np.linalg.norm(reconstructions[1]), 1.))
    unchanged = (protocol['source_hash'] == source_hash() and protocol['input_sha256'] == file_hash(path)
                 and protocol['runner_sha256'] == file_hash(__file__)
                 and all(protocol[key] == file_hash(Path(__file__).with_name(name)) for key, name in (
                     ('solver_sha256','scene_quadratic.py'), ('checkpoint_sha256','experiment_stages.py'),
                     ('compatibility_sha256','input_compatibility.py')))
                 and protocol['bridge_sha256'] == file_hash(BRIDGE))
    passed = len(rows) == 2 and all(r['converged'] for r in rows) and change is not None and change <= protocol['image_stability_tolerance'] and unchanged
    report = {'status': 'valid' if passed else 'incomplete', 'scope': protocol['decision'],
              'numerical_pilot_passed': passed, 'q3_authorized': False, 'source_input_unchanged': unchanged,
              'indices': indices.tolist(), 'scene_shape': list(shape), 'detector_shape': [cfg.eval_size]*2,
              'runs': rows, 'image_relative_change': change, 'wall_s': time.monotonic()-started,
              'process_peak_rss_bytes': peak_rss_bytes()}
    write_json(directory/f'attempt-{time.time_ns()}.json', report)
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--crop', choices=('feature', 'bland'), default='feature')
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--budgets', nargs=2, type=int, default=(100, 200))
    parser.add_argument('--ridge', type=float, default=.001)
    parser.add_argument('--noise', choices=('observed', 'expected'), default='observed')
    parser.add_argument('--weighting', choices=('scalar', 'spatial'), default='scalar')
    parser.add_argument('--wall-budget-s', type=float, default=300.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    report = run(args.input, args.out, crop=args.crop, frames=args.frames, budgets=args.budgets,
                 ridge=args.ridge, noise=args.noise, weighting=args.weighting,
                 resume=args.resume, wall_budget_s=args.wall_budget_s)
    sys.exit(0 if report['numerical_pilot_passed'] else 1)
