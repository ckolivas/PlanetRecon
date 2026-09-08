"""Bounded three-mode full-count cache profile with independent CPU products."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import ObservedSceneData, CellFFTBatch
from tools.scene_quadratic import SceneQuadratic
from tools.scene_parallel_reference import ParallelSceneBatch
from tools.scene_retained_cache import RetainedCellFFTBatch, spectrum_storage
from tools.study_io import identities, open_study, file_hash, write_json


def relative_error(actual, expected):
    value = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-30))
    return value if np.isfinite(value) else float('inf')


def run(path, directory, *, cropped=False):
    import torch
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    deps = [Path(__file__), *map(Path, ['tools/scene_study.py', 'tools/scene_fft.py',
            'tools/scene_quadratic.py', 'tools/scene_parallel_reference.py',
            'tools/scene_retained_cache.py', 'docs/scene-retained-cache-protocol.md'])]
    if cropped:
        deps += [Path('tools/scene_cropped_fft.py'),Path('docs/scene-cropped-fft.md'),Path('docs/scene-cropped-fft-protocol.md')]
    identity = identities(deps)
    modes = [('lru-256mib', 256*1024**2, CellFFTBatch),
             ('lru-3gib', 3*1024**3, CellFFTBatch),
             ('retain-first-3gib', 3*1024**3, RetainedCellFFTBatch)]
    if cropped:
        from tools.scene_cropped_fft import CroppedCellFFTBatch
        modes=[('full-3gib',3*1024**3,RetainedCellFFTBatch),
               ('crop-3gib',3*1024**3,CroppedCellFFTBatch),
               ('crop-4gib',4*1024**3,CroppedCellFFTBatch)]
    protocol = {'identities': identity, 'input_sha256': file_hash(path),
                'modes': [{'name': name, 'cache_bytes': budget} for name,budget,_ in modes],
                'count': 500, 'crop': 'feature', 'margin': 64, 'sum_native_strength': .0003,
                'threads': 2, 'cpu_frame_workers': 8, 'headroom_bytes': 1024**3,
                'wall_budget_s': 600, 'relative_tolerance': 1e-10,
                'probe': 'observed brightness times (1+0.25 Rademacher), seed 7171' if cropped else 'constant observed brightness',
                'scope': 'Cache retention/operator profile only; no reconstruction or scientific qualification.',
                'q3_authorized': False}
    directory = open_study(directory, protocol)
    start = time.monotonic(); rows = []; failures = []; cpu = None; storage = None
    def timed(fn):
        if time.monotonic()-start >= 600: raise TimeoutError('declared profile budget exhausted')
        before = time.monotonic(); value = fn()
        return value, time.monotonic()-before
    try:
        data, data_s = timed(lambda: ObservedSceneData(path, range(500), 'feature'))
        ops = data.operators(range(500), margin=64)
        images = [data.images[i] for i in range(500)]; variances = [data.variances[i] for i in range(500)]
        storage = spectrum_storage(ops[0].base.scene_shape, ops[0].base.psfs[0].shape, 500)
        reference_batch = ParallelSceneBatch(ops, workers=8)
        reference, setup_s = timed(lambda: SceneQuadratic(ops, images, variances, ridge=.0003/500, batch=reference_batch))
        x = np.full(reference.shape, max(float(np.mean(images)), 0.)/data.cfg.bin_factor**2)
        expected_forward=None;forward_s=None
        if cropped:
            x*=1+.25*np.random.default_rng(7171).choice([-1.,1.],size=x.shape)
            expected_forward,forward_s=timed(lambda: reference_batch.forward(x))
        expected, normal_s = timed(lambda: reference.normal(x))
        objective, objective_s = timed(lambda: reference.objective(x))
        cpu = {'data_psf_setup_s': data_s, 'quadratic_setup_s': setup_s, 'normal_s': normal_s,
               'objective_s': objective_s, 'forward_s':forward_s, 'workers': reference_batch.cache_info()}
        for name, budget, cls in modes:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            if free < budget+protocol['headroom_bytes']:
                raise MemoryError('insufficient declared headroom for '+name)
            batch = cls(ops, device='cuda', cache_bytes=budget)
            try:
                problem, setup_s = timed(lambda: SceneQuadratic(ops, images, variances, ridge=.0003/500, batch=batch))
                before = batch.cache_info(); times = []; errors = []
                for _ in range(3):
                    value, seconds = timed(lambda: problem.normal(x))
                    times.append(seconds); errors.append(relative_error(value, expected))
                result_objective, objective_s = timed(lambda: problem.objective(x))
                checks = {'normal': max(errors), 'linear': relative_error(problem.linear, reference.linear),
                          'objective': abs(result_objective-objective)/max(abs(objective), 1.)}
                forward_seconds=None
                if cropped:
                    predictions,forward_seconds=timed(lambda: batch.forward(x))
                    checks['maximum_frame_forward']=max(relative_error(a,b) for a,b in zip(predictions,expected_forward))
                row = {'mode': name, 'passed': all(np.isfinite(value) and value <= 1e-10 for value in checks.values()),
                       'relative_errors': checks, 'setup_s': setup_s, 'normal_s': times,
                       'objective_s': objective_s, 'forward_s':forward_seconds, 'cache_before_products': before,
                       'cache_after_products_and_objective': batch.cache_info(),
                       'gpu_free_before_bytes': free, 'gpu_total_bytes': total,
                       'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'process_peak_rss_bytes': peak_rss_bytes()}
                rows.append(row); write_json(directory/(name+'.json'), row)
                print(name, 'passed=', row['passed'], 'normal_s=', times, flush=True)
            finally:
                batch.clear_cache()
            del problem, batch
    except Exception as exc:
        failures.append(repr(exc)); print(repr(exc), flush=True)
    unchanged = identity == identities(deps) and file_hash(path) == protocol['input_sha256']
    passed = unchanged and not failures and len(rows) == 3 and all(r['passed'] for r in rows)
    report = {'status': 'valid' if passed else 'incomplete', 'source_input_unchanged': unchanged,
              'cpu': cpu, 'spectrum_storage': storage, 'rows': rows, 'failures': failures,
              'wall_s': time.monotonic()-start, 'scope': protocol['scope'], 'q3_authorized': False}
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cropped',action='store_true',help='Run the declared exact-crop padding comparison')
    a = p.parse_args(); sys.exit(0 if run(a.input,a.out,cropped=a.cropped)['status'] == 'valid' else 1)
