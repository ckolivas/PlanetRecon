"""Known-transfer selection-endpoint pilot with bounded, exact-identity stage resume."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import ObservedSceneData, CellFFTBatch, prior_coefficient
from tools.scene_quadratic import SceneQuadratic, solve
from tools.scene_iteration_state import IterationCheckpoint
from tools.experiment_stages import StageStore
from tools.study_io import identities, open_study, write_json, file_hash


def validate_case(manifest, seed, dr0, crop):
    if manifest.get('status') != 'valid' or not manifest.get('source_input_unchanged'):
        raise ValueError('valid unchanged selection manifest required')
    cases = [c for c in manifest['cases'] if (c['seed'], c['dr0'], c['crop']) == (seed, dr0, crop)]
    if len(cases) != 1:
        raise ValueError('exactly one matching selection case required')
    case = cases[0]
    from planetrecon.rank import subsets_from_scores
    scores = np.asarray(case['scores'])
    if scores.shape != (500,) or not np.isfinite(scores).all() or case['n_captured'] != 500:
        raise ValueError('500 finite observed ranking scores required')
    expected = subsets_from_scores(scores, (5, 10, 25, 50, 100))
    rows = case['selections']
    if [r['fraction'] for r in rows] != list(expected):
        raise ValueError('all five unique ordered selection fractions required')
    for row in rows:
        idx = expected[row['fraction']].tolist()
        if row['indices'] != idx or row['n_used'] != len(idx) or row['mean_ridge'] != .0003/len(idx):
            raise ValueError('selection indices/count/prior differ from frozen ranking')
    return case


def run(path, manifest_path, directory, *, device='cpu', resume=False):
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    deps = [Path(__file__), Path('tools/scene_study.py'), Path('tools/scene_fft.py'),
            Path('tools/scene_quadratic.py'), Path('tools/scene_iteration_state.py'),
            Path('docs/scene-full-selection-pilot-protocol.md')]
    identity = identities(deps)
    manifest = json.loads(manifest_path.read_text())
    selected = validate_case(manifest, 1001, 4., 'feature')
    if file_hash(path) != selected['input_sha256']:
        raise ValueError('capture does not match selection manifest')
    protocol = {'identities': identity, 'input_sha256': file_hash(path),
                'manifest_sha256': file_hash(manifest_path), 'case': [1001, 4., 'feature'],
                'fractions': [5, 100], 'sum_native_strength': .0003,
                'budgets': [750, 1500], 'tolerance': 1e-5, 'image_tolerance': 1e-4,
                'margin': 64, 'cell_factor': 1, 'device': device, 'threads': 2,
                'cache_bytes': 256*1024**2, 'fit_budget_s': 300., 'wall_budget_s': 1200.,
                'scope': 'One-case 5%/100% endpoint numerical pilot; no full family, scientific prior or Gate-1/Q2/Q3 qualification.',
                'q3_authorized': False}
    directory = open_study(directory, protocol, resume=resume)
    started = time.monotonic(); deadline = started+protocol['wall_budget_s']
    rows, failures = [], []
    try:
        data = ObservedSceneData(path, range(500), 'feature')
        for selection in selected['selections']:
            fraction = selection['fraction']; indices = selection['indices']
            if fraction not in protocol['fractions']:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError('pilot budget exhausted before fraction '+str(fraction))
            start = time.monotonic()
            ops = data.operators(indices, margin=64)
            images = [data.images[i] for i in indices]; variances = [data.variances[i] for i in indices]
            ridge = prior_coefficient(.0003, len(indices))
            batch = CellFFTBatch(ops, device=device, cache_bytes=protocol['cache_bytes'])
            problem = SceneQuadratic(ops, images, variances, ridge=ridge, batch=batch)
            reference = SceneQuadratic(ops, images, variances, ridge=ridge)
            store = StageStore(directory/'stages'/str(fraction), {'protocol': protocol, 'selection': selection}, deadline=deadline)
            fits, outputs = [], []
            for cap in protocol['budgets']:
                def compute():
                    checkpoint = IterationCheckpoint(directory/'iterations'/f'{fraction}-{cap}.npz',
                                                     {'protocol': protocol, 'selection': selection})
                    def progress(n, x, certificate):
                        write_json(directory/'progress.json', {'fraction': fraction, 'cap': cap,
                                   'iteration': n, 'certificate': certificate})
                    x, info = solve(problem, maxiter=cap, tolerance=protocol['tolerance'],
                                    deadline=min(deadline, time.monotonic()+protocol['fit_budget_s']),
                                    iteration_checkpoint=checkpoint, checkpoint_interval=100, callback=progress)
                    info['reference_certificate'] = reference.certificate(x)
                    if not info['converged'] and info['reason'] == 'wall_budget':
                        write_json(directory/f'incomplete-{fraction}-{cap}-{time.time_ns()}.json', info)
                        raise TimeoutError('fit deadline; exact iteration state retained')
                    return x, info
                try:
                    x, info = store.run(f'budget-{cap}', compute)
                    fits.append(info); outputs.append(x)
                except TimeoutError as exc:
                    fits.append({'maxiter': cap, 'converged': False, 'reason': str(exc)})
            changes = {}
            if len(outputs) == 2:
                changes['latent'] = float(np.linalg.norm(outputs[0]-outputs[1])/max(np.linalg.norm(outputs[1]), 1.))
                a, b = (ops[0].detector_scene(x) for x in outputs)
                changes['detector'] = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1.))
            passed = (len(outputs) == 2 and all(f['converged'] and f['reference_certificate']['feasible']
                      and f['reference_certificate']['relative_solution_error_bound'] <= protocol['tolerance'] for f in fits)
                      and max(changes.values()) <= protocol['image_tolerance'])
            row = {'fraction': fraction, 'indices': indices, 'n_used': len(indices), 'n_captured': 500,
                   'mean_ridge': ridge, 'observed_electron_sum': float(sum(y.sum() for y in images)),
                   'numerical_passed': bool(passed), 'runs': fits, 'relative_changes': changes,
                   'cache': batch.cache_info(), 'wall_s': time.monotonic()-start,
                   'process_peak_rss_bytes': peak_rss_bytes()}
            rows.append(row); write_json(directory/f'fraction-{fraction}.json', row)
            print(fraction, 'passed=', passed, 'iterations=', [f.get('n_iter') for f in fits], flush=True)
            batch.clear_cache()
            del batch, problem, reference, ops, outputs
    except Exception as exc:
        failures.append(repr(exc)); print(repr(exc), flush=True)
    unchanged = (identity == identities(deps) and file_hash(path) == protocol['input_sha256']
                 and file_hash(manifest_path) == protocol['manifest_sha256'])
    complete = [r['fraction'] for r in rows] == protocol['fractions'] and not failures
    report = {'status': 'valid' if complete and unchanged and all(r['numerical_passed'] for r in rows) else 'incomplete',
              'complete': complete, 'source_input_unchanged': unchanged, 'rows': rows, 'failures': failures,
              'wall_s': time.monotonic()-started, 'q3_authorized': False, 'scope': protocol['scope']}
    write_json(directory/f'attempt-{time.time_ns()}.json', report); write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    sys.exit(0 if run(a.input, a.manifest, a.out, device=a.device, resume=a.resume)['status'] == 'valid' else 1)
