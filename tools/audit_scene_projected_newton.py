"""Bounded endpoint comparison for a safeguarded Newton-CG update on the same scene objective."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.audit_scene_selections import validate_case
from tools.scene_study import ObservedSceneData, CellFFTBatch, prior_coefficient
from tools.scene_quadratic import SceneQuadratic
from tools.scene_projected_newton import solve
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.experiment_stages import StageStore
from tools.study_io import identities, open_study, write_json, file_hash


def run(path, manifest_path, directory, *, device='cpu', resume=False):
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    deps = [Path(__file__), Path('tools/audit_scene_selections.py'), Path('tools/scene_study.py'),
            Path('tools/scene_fft.py'), Path('tools/scene_quadratic.py'), Path('tools/scene_parallel_reference.py'),
            Path('tools/scene_projected_newton.py'), Path('docs/scene-projected-newton-protocol.md')]
    identity = identities(deps)
    selected = validate_case(json.loads(manifest_path.read_text()), 1001, 4., 'feature')
    if file_hash(path) != selected['input_sha256']:
        raise ValueError('capture does not match manifest')
    protocol = {'identities': identity, 'input_sha256': file_hash(path),
                'manifest_sha256': file_hash(manifest_path), 'case': [1001, 4., 'feature'],
                'fractions': [5, 100], 'sum_native_strength': .0003, 'margin': 64, 'cell_factor': 1,
                'budgets': [750, 1500], 'tolerance': 1e-5, 'image_tolerance': 1e-4,
                'device': device, 'threads': 2, 'cache_bytes': 256*1024**2,
                'fit_budget_s': 300., 'wall_budget_s': 1200., 'inner_steps': 32, 'max_line_search': 12, 'cpu_frame_workers': 8, 'budget_unit': 'inner/line-search/refresh Hessian products; initialization/final checks separate',
                'background_workload': 'One scientific study at a time; normal desktop activity may overlap; historical timings are not isolated speed comparisons.',
                'scope': 'Safeguarded Newton-CG numerical endpoint pilot; no full-family or scientific qualification; stage resume only.',
                'q3_authorized': False}
    directory = open_study(directory, protocol, resume=resume)
    started = time.monotonic(); deadline = started+protocol['wall_budget_s']
    rows, failures = [], []
    try:
        data = ObservedSceneData(path, range(500), 'feature')
        for selection in selected['selections']:
            fraction = selection['fraction']; indices = selection['indices']
            if fraction not in protocol['fractions']: continue
            if time.monotonic() >= deadline: raise TimeoutError('pilot budget exhausted before fraction '+str(fraction))
            start = time.monotonic()
            ops = data.operators(indices, margin=64)
            images = [data.images[i] for i in indices]; variances = [data.variances[i] for i in indices]
            ridge = prior_coefficient(.0003, len(indices))
            batch = CellFFTBatch(ops, device=device, cache_bytes=protocol['cache_bytes'])
            problem = SceneQuadratic(ops, images, variances, ridge=ridge, batch=batch)
            reference_batch = ParallelSceneBatch(ops, workers=8)
            reference = SceneQuadratic(ops, images, variances, ridge=ridge, batch=reference_batch)
            store = StageStore(directory/'stages'/str(fraction), {'protocol': protocol, 'selection': selection}, deadline=deadline)
            fits, outputs = [], []
            for cap in protocol['budgets']:
                def compute():
                    trace = []
                    def progress(row, x):
                        trace.append(row)
                        write_json(directory/f'trace-{fraction}-{cap}.json', {'trace': trace})
                    x, info = solve(problem, max_products=cap, inner_steps=32,
                                    tolerance=protocol['tolerance'], callback=progress,
                                    deadline=min(deadline, time.monotonic()+protocol['fit_budget_s']))
                    info['reference_certificate'] = reference.certificate(x)
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
                   'cache': batch.cache_info(), 'cpu_reference': reference_batch.cache_info(), 'wall_s': time.monotonic()-start,
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
