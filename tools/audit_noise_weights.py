"""Full-grid known-transfer sensitivity to oracle per-pixel noise weights.

This is an unconstrained quadratic diagnostic, not a new production likelihood.
Variance is frozen from simulator expectations, never fitted from noisy pixels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, cg

from tools.audit_full_gate1 import complete_family, file_hash, write_json


def weighted_solve(otfs, images, variance, lam, *, maxiter=256, tol=1e-9, batch=32):
    """Solve (sum H* W H + L)o=sum H* W y with a Fourier preconditioner."""
    if maxiter < 1 or batch < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('positive budgets and tolerance required')
    if (otfs.shape != images.shape or variance.shape != images.shape
            or lam.shape != images.shape[-2:] or np.any(variance <= 0)
            or not np.isfinite(variance).all() or not np.isfinite(lam).all() or np.any(lam <= 0)):
        raise ValueError('matching images/transfers/positive variance and positive regularisation required')
    shape = images.shape[-2:]
    n = int(np.prod(shape))
    weights = 1/variance
    rhs_f = np.zeros(shape, dtype=complex)
    preconditioner = lam.copy()
    for start in range(0, len(images), batch):
        sl = slice(start, start+batch)
        rhs_f += np.sum(otfs[sl].conj()*fft2(weights[sl]*images[sl]), axis=0)
        preconditioner += np.sum(abs(otfs[sl])**2 * weights[sl].mean(axis=(-2, -1))[:, None, None], axis=0)
    rhs = ifft2(rhs_f).real.ravel()

    def normal(vec):
        spectrum = fft2(vec.reshape(shape))
        result = lam*spectrum
        for start in range(0, len(images), batch):
            sl = slice(start, start+batch)
            predicted = ifft2(otfs[sl]*spectrum).real
            result += np.sum(otfs[sl].conj()*fft2(weights[sl]*predicted), axis=0)
        return ifft2(result).real.ravel()

    def precondition(vec):
        return ifft2(fft2(vec.reshape(shape))/preconditioner).real.ravel()

    iterations = 0

    def count(_):
        nonlocal iterations
        iterations += 1

    solution, code = cg(LinearOperator((n, n), matvec=normal, dtype=float), rhs,
                        M=LinearOperator((n, n), matvec=precondition, dtype=float),
                        x0=np.zeros(n), rtol=tol, atol=0., maxiter=maxiter, callback=count)
    residual = float(np.linalg.norm(normal(solution)-rhs)/max(np.linalg.norm(rhs), 1e-30))
    return solution.reshape(shape), {'cg_info': int(code), 'n_iter': iterations, 'maxiter': maxiter,
                                     'normal_relative_residual': residual, 'tolerance': tol,
                                     'converged': bool(code == 0 and residual <= tol)}


def case(payload):
    from planetrecon.evaluate import REG, _metrics, load_crop
    from planetrecon.estimators import e1, frame_noise_variance, register_images
    from planetrecon.hdf5io import gate_errors
    from planetrecon.physics_audit import peak_rss_bytes
    from planetrecon.rank import score_sequence, subsets_from_scores
    from planetrecon.runtime import apply_thread_limits
    path, crop_name, directory = payload
    apply_thread_limits(2)
    if errors := gate_errors(path):
        raise ValueError(f'{path}: {errors}')
    identity = file_hash(path)
    cfg, crop, extras = load_crop(path, crop_name)
    if cfg.n_frames != 500 or cfg.eval_size != 128:
        raise ValueError('full-resolution 500-frame inputs required')
    started = time.monotonic()
    sigma2 = frame_noise_variance(crop.expected, extras['read_noise_e'])
    variance = np.maximum(crop.expected, 0)+extras['read_noise_e']**2
    lam = REG.field_e1(cfg, cfg.eval_size)
    scores = score_sequence(register_images(crop.observed, crop.shifts), sky_mask=crop.sky_mask)
    subsets = subsets_from_scores(scores, (10, 100))
    rows = []
    for p, indices in subsets.items():
        h, observed, v = crop.otf[indices], crop.observed[indices], variance[indices]
        scalar = e1(h, observed, sigma2[indices], lam)
        fits, images = [], []
        for budget in (256, 512):
            t0 = time.monotonic()
            image, info = weighted_solve(h, observed, v, lam, maxiter=budget)
            images.append(image)
            fits.append({'solver': info, 'wall_s': time.monotonic()-t0, 'metrics': _metrics(image, crop)})
        change = float(np.linalg.norm(images[0]-images[1])/max(np.linalg.norm(images[1]), 1e-12))
        expected = crop.expected[indices]
        model = ifft2(h*fft2(crop.truth_e)).real
        rows.append({'p': p, 'indices': indices.tolist(), 'n_used': len(indices), 'n_captured': cfg.n_frames,
                     'photons_used': float(expected.sum()), 'scalar_metrics': _metrics(scalar, crop),
                     'weighted_fits': fits, 'budget_image_relative_change': change,
                     'scalar_weighted_image_relative_difference': float(np.linalg.norm(scalar-images[-1])/max(np.linalg.norm(scalar), 1e-12)),
                     'known_noise_mean_standardized_square': float(np.mean((observed-expected)**2/v)),
                     'truth_forward_mismatch_mean_standardized_square': float(np.mean((model-expected)**2/v)),
                     'variance_relative_rms': float(np.sqrt(np.mean((v/sigma2[indices, None, None]-1)**2))),
                     'converged': bool(all(f['solver']['converged'] for f in fits) and change <= 1e-3)})
    unchanged = identity == file_hash(path)
    row = {'seed': cfg.seed, 'dr0': cfg.dr0, 'crop': crop_name, 'config': asdict(cfg),
           'input_path': str(path), 'input_sha256': identity, 'input_unchanged': unchanged,
           'subsets': rows, 'numerical_controls_passed': unchanged and all(r['converged'] for r in rows),
           'wall_s': time.monotonic()-started, 'process_peak_rss_bytes': peak_rss_bytes()}
    dest = directory/f'case-{cfg.seed}-{int(cfg.dr0)}-{crop_name}.json'
    write_json(dest, row)
    return {k: row[k] for k in ('seed', 'dr0', 'crop', 'numerical_controls_passed', 'wall_s')} | {
        'path': dest.name, 'sha256': file_hash(dest)}


def run(inputs, directory, workers=2):
    from planetrecon import constants as C
    from planetrecon.evaluate import REG, family_paths
    from planetrecon.provenance import source_hash
    if not 1 <= workers <= 16:
        raise ValueError('workers must be 1–16')
    directory.mkdir(parents=True, exist_ok=False)
    source, runner = source_hash(), file_hash(__file__)
    write_json(directory/'protocol.json', {
        'seeds': C.DEV_SEEDS, 'regimes': C.MANDATORY_DR0, 'crops': ['feature', 'bland'],
        'n_frames': 500, 'eval_size': 128, 'subsets': [10, 100], 'budgets': [256, 512],
        'solver_tolerance': 1e-9, 'image_tolerance': 1e-3, 'regularisation': asdict(REG),
        'source_hash': source, 'runner_sha256': runner, 'q3_authorized': False,
        'design': 'Unconstrained known-transfer E1, fixed regularisation and frame membership. '
                  'Scalar frame-mean Poisson+read variance versus oracle per-pixel expected+read variance. '
                  'Zero-start preconditioned CG independently checked at doubled budget. '
                  'No claim of Poisson likelihood calibration, positivity/support or MFBD qualification.'})
    rows, failures = [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(case, (path, crop, directory)): (path, crop)
                   for path in family_paths(inputs, C.DEV_SEEDS) for crop in ('feature', 'bland')}
        for future in as_completed(pending):
            try:
                row = future.result()
                rows.append(row)
                print(f'{row["seed"]} {row["dr0"]} {row["crop"]}: converged={row["numerical_controls_passed"]}; {row["wall_s"]:.1f}s', flush=True)
            except Exception as exc:
                path, crop = pending[future]
                failures.append({'input': str(path), 'crop': crop, 'error': repr(exc)})
                print(f'{path.name} {crop}: {exc}', flush=True)
            write_json(directory/'progress.json', {'cases': rows, 'failures': failures})
    unchanged = source == source_hash() and runner == file_hash(__file__)
    complete = complete_family(rows, C.DEV_SEEDS, C.MANDATORY_DR0) and not failures
    report = {'status': 'diagnostic', 'q3_authorized': False, 'source_hash': source, 'source_unchanged': unchanged,
              'complete': complete, 'cases': sorted(rows, key=lambda r: (r['seed'], r['dr0'], r['crop'])),
              'failures': failures, 'wall_s': time.monotonic()-started,
              'numerical_controls_passed': complete and unchanged and all(r['numerical_controls_passed'] for r in rows)}
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    report = run(args.inputs, args.out, args.workers)
    sys.exit(0 if report['numerical_controls_passed'] else 1)
