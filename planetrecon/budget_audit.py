"""Fixed-model physical-screen development convergence and resource audit."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from planetrecon import constants as C
from planetrecon.config import make_config
from planetrecon.evaluate import load_crop, _to_jsonable
from planetrecon.operators import scalar_noise_mismatch
from planetrecon.provenance import sha256_bytes, source_hash
from planetrecon.q2 import evaluate_q2_crop
from planetrecon.simulate import simulate
from planetrecon.validate import check_lowfreq_ranking
from planetrecon.physics_audit import peak_rss_bytes


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    initial_source_hash = source_hash()
    protocol = {'status': 'diagnostic', 'q3_authorized': False, 'seeds': list(C.DEV_SEEDS),
                'regimes': [4., 8.], 'crops': ['feature', 'bland'], 'n_frames': 8,
                'n_diam': 16, 'eval_size': 32, 'modes': [15, 35, 60],
                'budgets': [{'outer': 12, 'phase': 96}, {'outer': 36, 'phase': 192}],
                'phase_initialization': 'zero versus subset with seeded 0.1 rad RMS higher phases',
                'source_hash': initial_source_hash,
                'inits': ['zero', 'subset'], 'image_stability_tolerance': 1e-3,
                'closure_stability_tolerance': .01, 'phase_relative_gradient_tolerance': C.Q2_PHASE_GRAD_REL_TOL,
                'joint_relative_change_tolerance': C.Q2_JOINT_REL_TOL,
                'object_solver_tolerance': C.E2A_FISTA_TOL, 'object_iteration_cap': C.E2A_FISTA_MAXITER,
                'partition': '6 training, 1 selection, 1 assessment; fixed before fits',
                'limits': 'Reduced grids and frame count; fixed regularizer and snapshot model; no full-resolution gate qualification.'}
    (directory/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    cases = []
    for seed in protocol['seeds']:
        for regime in protocol['regimes']:
            cfg = make_config(seed, regime, n_frames=8, n_diam=16, pupil_pad_factor=8., eval_size=32)
            path = simulate(cfg, directory/'inputs')
            input_hash = sha256_bytes(path.read_bytes())
            for crop_name in protocol['crops']:
                cfg, crop, extras = load_crop(path, crop_name)
                images = []
                rows = []
                for budget in protocol['budgets']:
                    print(f"seed={seed} D/r0={regime} {crop_name} budget={budget}", flush=True)
                    t0 = time.monotonic()
                    fit = evaluate_q2_crop(cfg, crop, extras, holdout=True, tv_mu=0.,
                        m_grid=(15, 35, 60), outer_iters=(budget['outer'],)*3, alpha_iters=budget['phase'],
                        frame_workers=2, return_reconstructions=True)
                    recon = fit.pop('_reconstructions')
                    images.append(recon)
                    # Compact synthetic images remain local; the public report
                    # records numerical comparisons and their array identities.
                    filename = f"recon-{seed}-{int(regime)}-{crop_name}-{budget['outer']}.npz"
                    np.savez_compressed(directory/filename, **recon)
                    rows.append({'budget': budget, 'wall_s': time.monotonic()-t0, 'fit': fit,
                                 'cumulative_process_peak_rss_bytes': peak_rss_bytes(),
                                 'reconstruction_hashes': {k: sha256_bytes(v.tobytes()) for k, v in recon.items()}})
                    print(f"  {fit['status']}; assessment={fit['assessment']['status']}; {rows[-1]['wall_s']:.2f}s", flush=True)
                stability = {k: float(np.linalg.norm(images[1][k]-images[0][k])/max(np.linalg.norm(images[1][k]), 1e-12))
                             for k in images[0]}
                cross_start = {str(b['outer']): float(np.linalg.norm(im['zero']-im['subset'])/max(np.linalg.norm(im['zero']), 1e-12))
                               for b, im in zip(protocol['budgets'], images)}
                closure_delta = abs(rows[1]['fit']['C']-rows[0]['fit']['C'])
                record = {'seed': seed, 'dr0': regime, 'crop': crop_name, 'input_sha256': input_hash,
                          'captured_frames': 8, 'used_frames_all_frame_fit': 8,
                          'expected_photons_all_frames': float(crop.expected.sum()),
                          'noise_maps': [scalar_noise_mismatch(im, extras['read_noise_e']) for im in crop.expected],
                          'budgets': rows, 'image_relative_changes': stability,
                          'cross_start_relative_difference': cross_start, 'closure_absolute_change': closure_delta,
                          'bounded_convergence_passed': bool(all(row['fit']['status'] == 'valid' for row in rows)
                              and max(stability.values()) <= protocol['image_stability_tolerance']
                              and max(cross_start.values()) <= protocol['image_stability_tolerance']
                              and closure_delta <= protocol['closure_stability_tolerance'])}
                dest = directory/f'case-{seed}-{int(regime)}-{crop_name}.json'
                dest.write_text(json.dumps(_to_jsonable(record), indent=2, allow_nan=False)+'\n')
                cases.append({'seed': seed, 'dr0': regime, 'crop': crop_name, 'path': dest.name,
                              'sha256': sha256_bytes(dest.read_bytes()), 'bounded_convergence_passed': record['bounded_convergence_passed'],
                              'max_image_relative_change': max(stability.values()), 'closure_absolute_change': closure_delta,
                              'wall_s': sum(r['wall_s'] for r in rows)})
    ranking = []
    for seed in protocol['seeds']:
        for regime in protocol['regimes']:
            cfg = make_config(seed, regime, n_diam=16, pupil_pad_factor=8., n_frames=1)
            ranking.append({'seed': seed, 'dr0': regime, 'checks': [asdict(c) for c in check_lowfreq_ranking(cfg)]})
    final_source_hash = source_hash()
    report = {'status': 'diagnostic', 'q3_authorized': False, 'source_hash': initial_source_hash,
              'source_hash_after': final_source_hash, 'source_unchanged': initial_source_hash == final_source_hash,
              'cases': cases, 'low_frequency_ranking': ranking, 'wall_s': time.monotonic()-started,
              'complete': len(cases) == 12, 'all_bounded_checks_passed': initial_source_hash == final_source_hash and all(c['bounded_convergence_passed'] for c in cases)}
    (directory/'report.json').write_text(json.dumps(_to_jsonable(report), indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    import argparse
    from planetrecon.runtime import apply_thread_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    apply_thread_limits(2)
    run(args.out)
