"""Profile bounded CPU verification using the original spatial frame operators."""
import argparse
from pathlib import Path
import sys
import time
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import ObservedSceneData, prior_coefficient
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.study_io import identities, file_hash, open_study, write_json


def run(path, directory):
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    old_report = Path('results/p5-full-count-profile/report.json')
    deps = [Path(__file__), Path('tools/scene_study.py'), Path('tools/scene_fft.py'),
            Path('tools/scene_parallel_reference.py'), Path('docs/scene-parallel-reference-protocol.md')]
    identity = identities(deps)
    protocol = {'identities': identity, 'input_sha256': file_hash(path), 'old_report_sha256': file_hash(old_report),
                'count': 500, 'workers': [1, 8], 'fft_threads_per_worker': 1, 'blas_threads': 2,
                'wall_budget_s': 600., 'margin': 64, 'sum_native_strength': .0003,
                'background_workload': 'May overlap both endpoint pilots; no isolated speed benchmark.',
                'q3_authorized': False}
    directory = open_study(directory, protocol)
    start = time.monotonic(); rows, failures = [], []
    def timed(fn):
        if time.monotonic()-start >= 600: raise TimeoutError('CPU profile budget exhausted')
        t = time.monotonic(); result = fn()
        return result, time.monotonic()-t
    try:
        data, setup_s = timed(lambda: ObservedSceneData(path, range(500)))
        ops = data.operators(range(500), margin=64)
        images = list(data.images.values())
        weights = [np.full(o.output_shape, 1/data.variances[i]/500) for i, o in enumerate(ops)]
        residuals = [w*y for w, y in zip(weights, images)]
        x = np.full(ops[0].scene_shape, max(float(np.mean(images)), 0.)/data.cfg.bin_factor**2)
        ridge = prior_coefficient(.0003, 500)
        previous = next(r for r in json.loads(old_report.read_text())['rows'] if r['count'] == 500)
        ref_normal = ref_linear = None
        for workers in (1, 8):
            batch = ParallelSceneBatch(ops, workers=workers)
            normal, normal_s = timed(lambda: batch.normal(x, weights))
            linear, linear_s = timed(lambda: batch.adjoint(residuals))
            if workers == 1: ref_normal, ref_linear = normal, linear
            gradient = normal+ridge*x-linear
            residual = np.where(x > 0, gradient, np.minimum(gradient, 0.))
            bound = float(np.linalg.norm(residual)/ridge/max(np.linalg.norm(x), 1.))
            old_bound = previous['probe_certificate']['relative_solution_error_bound']
            bound_difference = abs(bound-old_bound)/max(abs(old_bound), 1e-30)
            row = {'workers': workers, 'normal_s': normal_s, 'linear_s': linear_s,
                   'data_psf_setup_s': setup_s, 'normal_bitwise_equal': np.array_equal(normal, ref_normal),
                   'linear_bitwise_equal': np.array_equal(linear, ref_linear),
                   'relative_distance_bound': bound, 'previous_bound_relative_difference': bound_difference,
                   'pending_frames': batch.cache_info(), 'process_peak_rss_bytes': peak_rss_bytes()}
            row['passed'] = row['normal_bitwise_equal'] and row['linear_bitwise_equal'] and bound_difference <= 1e-10
            rows.append(row); write_json(directory/f'workers-{workers}.json', row)
            print(workers, 'passed=', row['passed'], 'normal_s=', normal_s, 'linear_s=', linear_s, flush=True)
    except Exception as exc:
        failures.append(repr(exc)); print(repr(exc), flush=True)
    unchanged = (identity == identities(deps) and file_hash(path) == protocol['input_sha256']
                 and file_hash(old_report) == protocol['old_report_sha256'])
    report = {'status': 'valid' if len(rows) == 2 and not failures and unchanged and all(r['passed'] for r in rows) else 'incomplete',
              'source_input_unchanged': unchanged, 'rows': rows, 'failures': failures,
              'wall_s': time.monotonic()-start, 'q3_authorized': False,
              'scope': 'Parallel independent CPU verifier kernel profile; constant probe, no reconstruction or scientific gate.'}
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.exit(0 if run(a.input, a.out)['status'] == 'valid' else 1)
