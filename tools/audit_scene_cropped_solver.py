"""Reference/diagonal Newton endpoints under the qualified exact-crop FFT backend."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.audit_scene_selections import validate_case
from tools.scene_study import ObservedSceneData, prior_coefficient
from tools.scene_cropped_fft import CroppedCellFFTBatch
from tools.scene_iteration_state import IterationCheckpoint
from tools.scene_quadratic import SceneQuadratic, solve as reference_solve
from tools.scene_projected_newton import solve as newton_solve
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.experiment_stages import StageStore
from tools.study_io import identities, open_study, write_json, file_hash


def run(path, manifest_path, directory, *, method='newton', resume=False):
    if method not in ('newton','reference'): raise ValueError('declared method required')
    device='cuda'
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    deps = [Path(__file__), Path('tools/audit_scene_selections.py'), Path('tools/scene_study.py'),
            Path('tools/scene_fft.py'), Path('tools/scene_quadratic.py'), Path('tools/scene_parallel_reference.py'),
            Path('tools/scene_projected_newton.py'), Path('tools/scene_iteration_state.py'),
            Path('tools/scene_retained_cache.py'),Path('tools/scene_cropped_fft.py'),
            Path('docs/scene-cropped-solver-protocol.md')]
    identity = identities(deps)
    selected = validate_case(json.loads(manifest_path.read_text()), 1001, 4., 'feature')
    if file_hash(path) != selected['input_sha256']:
        raise ValueError('capture does not match manifest')
    evidence_path=Path('results/p5-cropped-fft/report.json')
    evidence_protocol=evidence_path.with_name('protocol.json')
    evidence=json.loads(evidence_path.read_text()); qualified=json.loads(evidence_protocol.read_text())
    if evidence['status']!='valid' or not evidence['source_input_unchanged'] or qualified['input_sha256']!=file_hash(path):
        raise ValueError('qualified cropped FFT evidence/input required')
    if qualified['identities']['package_source_hash']!=identity['package_source_hash']:
        raise ValueError('qualified package source changed')
    for name,digest in qualified['identities']['dependencies'].items():
        if file_hash(Path(name))!=digest: raise ValueError('qualified numerical source changed: '+name)
    evidence_hashes={str(p):file_hash(p) for p in (evidence_path,evidence_protocol)}
    protocol = {'identities': identity,'method':method,'qualification_hashes':evidence_hashes, 'input_sha256': file_hash(path),
                'manifest_sha256': file_hash(manifest_path), 'case': [1001, 4., 'feature'],
                'fractions': [5, 100], 'sum_native_strength': .0003, 'margin': 64, 'cell_factor': 1,
                'budgets': [750, 1500], 'tolerance': 1e-5, 'image_tolerance': 1e-4,
                'device': device, 'threads': 2, 'cache_bytes': 4*1024**3,'headroom_bytes':1024**3,
                'fit_budget_s': {'5':300.,'100':900.}, 'wall_budget_s': 2400., 'inner_steps': 32, 'max_line_search': 12, 'cpu_frame_workers': 8, 'budget_unit': 'iterations' if method=='reference' else 'inner/line-search/refresh Hessian products; initialization/final checks separate',
                'background_workload': 'One scientific study at a time; normal desktop activity may overlap; historical timings are not isolated speed comparisons.',
                'scope': 'Qualified cropped-FFT '+method+' endpoint pilot; no full-family or scientific qualification; completed-stage resume only.',
                'q3_authorized': False}
    directory = open_study(directory, protocol, resume=resume)
    started = time.monotonic(); deadline = started+protocol['wall_budget_s']
    rows, failures = [], []; batch=None
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
            import torch
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0]<protocol['cache_bytes']+protocol['headroom_bytes']:
                raise MemoryError('insufficient declared GPU headroom')
            batch = CroppedCellFFTBatch(ops, device=device, cache_bytes=protocol['cache_bytes'])
            problem = SceneQuadratic(ops, images, variances, ridge=ridge, batch=batch)
            reference_batch = ParallelSceneBatch(ops, workers=8)
            reference = SceneQuadratic(ops, images, variances, ridge=ridge, batch=reference_batch)
            store = StageStore(directory/'stages'/str(fraction), {'protocol': protocol, 'selection': selection}, deadline=deadline)
            fits, outputs = [], []
            for cap in protocol['budgets']:
                def compute():
                    trace = [];inner_trace=[]
                    def progress(row, x):
                        trace.append(row)
                        write_json(directory/f'trace-{fraction}-{cap}.json', {'trace': trace})
                    fit_deadline=min(deadline,time.monotonic()+protocol['fit_budget_s'][str(fraction)])
                    if method=='newton':
                        def inner_progress(row):
                            inner_trace.append(row)
                            write_json(directory/f'inner-trace-{fraction}-{cap}.json',{'trace':inner_trace})
                        x,info=newton_solve(problem,max_products=cap,inner_steps=32,
                                            tolerance=protocol['tolerance'],callback=progress,
                                            inner_callback=inner_progress,deadline=fit_deadline)
                    else:
                        checkpoint=IterationCheckpoint(directory/'iterations'/f'{fraction}-{cap}.npz',
                                                       {'protocol':protocol,'selection':selection})
                        def reference_progress(n,x,cert):
                            progress({'iteration':n,**cert},x)
                        x,info=reference_solve(problem,maxiter=cap,tolerance=protocol['tolerance'],
                                              callback=reference_progress,deadline=fit_deadline,
                                              iteration_checkpoint=checkpoint,checkpoint_interval=100)
                        info['trace']=trace
                    info['reference_certificate'] = reference.certificate(x)
                    return x, info
                try:
                    x, info = store.run(f'budget-{cap}', compute)
                    fits.append(info); outputs.append(x)
                    print(method,fraction,cap,'bound=',info['reference_certificate']['relative_solution_error_bound'],flush=True)
                except TimeoutError as exc:
                    fits.append({'max_products': cap, 'converged': False, 'reason': str(exc)})
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
            del problem, reference, ops, outputs
            batch=None
    except Exception as exc:
        failures.append(repr(exc)); print(repr(exc), flush=True)
    finally:
        if batch is not None: batch.clear_cache()
    unchanged = (all(file_hash(Path(p))==h for p,h in evidence_hashes.items()) and identity == identities(deps) and file_hash(path) == protocol['input_sha256']
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
    p.add_argument('--out', type=Path, required=True); p.add_argument('--method', choices=('newton', 'reference'), default='newton')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    sys.exit(0 if run(a.input, a.manifest, a.out, method=a.method, resume=a.resume)['status'] == 'valid' else 1)
