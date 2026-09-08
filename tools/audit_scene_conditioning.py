"""Read-only diagnostic of a preserved incomplete full-frame scene iterate."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import ObservedSceneData, CellFFTBatch
from tools.scene_quadratic import SceneQuadratic
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.scene_iteration_state import IterationCheckpoint
from tools.scene_conditioning import cone_residual, correction_bound, probe_cg
from tools.audit_scene_selections import validate_case
from tools.experiment_stages import atomic_write
from tools.study_io import identities, open_study, file_hash, write_json


def frozen_checkpoint(path, original_protocol, selection):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
    expected = {'context': {'protocol': original_protocol, 'selection': selection},
                'maxiter': 750, 'scaling': 'diagonal', 'shape': [1024, 912],
                'solver': 'extended_scene_projected_acceleration_v4', 'tolerance': 1e-5}
    if metadata['identity'] != expected:
        raise ValueError('checkpoint does not match the frozen reference objective')
    state = IterationCheckpoint(path, expected).load()
    if state['iteration'] != 142 or state['x'].shape != (1024, 912):
        raise ValueError('expected the preserved 142-iteration full-frame checkpoint')
    return state


def run(path, checkpoint, directory, *, device='cuda'):
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    original_path = Path('results/p2-selection-endpoints-reference/protocol.json')
    manifest_path = Path('results/p2-full-selection-manifest/report.json')
    original = json.loads(original_path.read_text())
    selected = validate_case(json.loads(manifest_path.read_text()), 1001, 4., 'feature')['selections'][-1]
    if file_hash(path) != original['input_sha256'] or file_hash(manifest_path) != original['manifest_sha256']:
        raise ValueError('frozen input/manifest mismatch')
    # Do not import an old state across unreviewed numerical/physics changes.
    for name, expected in original['identities']['dependencies'].items():
        if file_hash(Path(name)) != expected: raise ValueError('original numerical dependency changed: '+name)
    deps = [Path(__file__), Path('tools/scene_study.py'), Path('tools/scene_fft.py'),
            Path('tools/scene_quadratic.py'), Path('tools/scene_parallel_reference.py'),
            Path('tools/scene_iteration_state.py'), Path('tools/scene_conditioning.py'),
            Path('tools/audit_scene_selections.py'), Path('docs/scene-conditioning-protocol.md')]
    identity = identities(deps)
    if identity['package_source_hash'] != original['identities']['package_source_hash']:
        raise ValueError('original package source changed')
    state = frozen_checkpoint(checkpoint, original, selected)
    protocol = {'identities': identity, 'input_sha256': file_hash(path),
                'checkpoint_sha256': file_hash(checkpoint), 'original_protocol_sha256': file_hash(original_path),
                'manifest_sha256': file_hash(manifest_path), 'initial_iteration': 142,
                'modes': ['identity', 'majorizer'], 'probe_steps': 32, 'wall_budget_s': 600.,
                'threads': 2, 'cpu_frame_workers': 8, 'device': device, 'cache_bytes': 256*1024**2,
                'product_relative_tolerance': 1e-10, 'original_distance_tolerance': 1e-5,
                'scene_updated': False, 'q3_authorized': False,
                'scope': 'Frozen-iterate conditioning/certificate diagnostic; does not reclassify historical solves.'}
    directory = open_study(directory, protocol)
    started = time.monotonic(); deadline = started+600; rows, failures = [], []; initial = None
    try:
        data = ObservedSceneData(path, range(500), 'feature')
        ops = data.operators(range(500), margin=64)
        images = [data.images[i] for i in range(500)]; variances = [data.variances[i] for i in range(500)]
        fast_batch = CellFFTBatch(ops, device=device, cache_bytes=protocol['cache_bytes'])
        problem = SceneQuadratic(ops, images, variances, ridge=selected['mean_ridge'], batch=fast_batch)
        reference_batch = ParallelSceneBatch(ops, workers=8)
        reference = SceneQuadratic(ops, images, variances, ridge=selected['mean_ridge'], batch=reference_batch)
        x = state['x']; gradient = reference.gradient(x); residual = cone_residual(x, gradient)
        fast_gradient = problem.gradient(x)
        gradient_error = float(np.linalg.norm(fast_gradient-gradient)/max(np.linalg.norm(gradient), 1e-30))
        raw = reference.certificate(x, gradient=gradient)
        initial = {'raw_certificate': raw, 'gradient_relative_error': gradient_error,
                   'active_fraction': float(np.mean(x == 0)), 'ridge': problem.ridge,
                   'majorizer_min': float(problem.majorizer.min()), 'majorizer_max': float(problem.majorizer.max()),
                   'scaled_condition_number_upper_bound': float(problem.majorizer.max()/problem.ridge),
                   'setup_wall_s': time.monotonic()-started}
        write_json(directory/'initial.json', initial)
        if gradient_error > 1e-10: raise ValueError('independent gradient parity failed')
        for mode in protocol['modes']:
            trace = []
            def callback(row, y):
                trace.append(row)
                write_json(directory/f'trace-{mode}.json', {'mode': mode, 'trace': trace,
                                                          'recursive_residuals_are_certificates': False})
            diagonal = np.ones(problem.shape) if mode == 'identity' else problem.majorizer
            y, info = probe_cg(problem.normal, residual, diagonal, steps=32, deadline=deadline, callback=callback)
            stream = BytesIO(); np.savez_compressed(stream, correction=y)
            atomic_write(directory/f'correction-{mode}.npz', stream.getvalue())
            check_started = time.monotonic()
            hy = reference.normal(y)
            gpu_hy = problem.normal(y)
            product_error = float(np.linalg.norm(hy-gpu_hy)/max(np.linalg.norm(hy), 1e-30))
            bound = correction_bound(x, gradient, y, hy, problem.ridge)
            row = {'mode': mode, 'probe': info, 'bound': bound,
                   'product_relative_error': product_error, 'independent_check_s': time.monotonic()-check_started,
                   'probe_complete': info['reason'] != 'wall_budget',
                   'would_meet_distance_tolerance': bound['relative_solution_error_bound'] <= 1e-5,
                   'cache': fast_batch.cache_info(), 'cpu_reference': reference_batch.cache_info(),
                   'process_peak_rss_bytes': peak_rss_bytes()}
            rows.append(row); write_json(directory/f'{mode}.json', row)
            print(mode, 'steps=', info['iterations'], 'bound=', bound['relative_solution_error_bound'],
                  'product_error=', product_error, flush=True)
        fast_batch.clear_cache()
    except Exception as exc:
        failures.append(repr(exc)); print(repr(exc), flush=True)
    unchanged = (identity == identities(deps) and file_hash(path) == protocol['input_sha256']
                 and file_hash(checkpoint) == protocol['checkpoint_sha256']
                 and file_hash(original_path) == protocol['original_protocol_sha256']
                 and file_hash(manifest_path) == protocol['manifest_sha256'])
    passed = unchanged and not failures and len(rows) == 2 and all(r['probe_complete'] and r['product_relative_error'] <= 1e-10 for r in rows)
    report = {'status': 'valid' if passed else 'incomplete', 'source_input_unchanged': unchanged,
              'initial': initial, 'rows': rows, 'failures': failures, 'wall_s': time.monotonic()-started,
              'scene_updated': False, 'q3_authorized': False, 'scope': protocol['scope']}
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    sys.exit(0 if run(a.input, a.checkpoint, a.out, device=a.device)['status'] == 'valid' else 1)
