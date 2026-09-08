"""Bounded full-frame resource and CPU/CUDA parity profile, without reconstruction."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.scene_study import ObservedSceneData, CellFFTBatch, prior_coefficient
from tools.scene_quadratic import SceneQuadratic
from tools.study_io import identities, open_study, file_hash, write_json


def frame_indices(count, captured=500):
    if int(count) != count or not 1 <= count <= captured:
        raise ValueError('count must be within captured frame range')
    return np.linspace(0, captured-1, int(count), dtype=int).tolist()


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b), 1e-30))


def run(path, directory):
    import torch
    from planetrecon.runtime import apply_thread_limits
    from planetrecon.physics_audit import peak_rss_bytes
    apply_thread_limits(2)
    deps = [Path(__file__), Path('tools/scene_study.py'), Path('tools/scene_fft.py'),
            Path('tools/scene_quadratic.py'), Path('docs/scene-full-count-profile-protocol.md')]
    identity = identities(deps)
    protocol = {'identities': identity, 'input_sha256': file_hash(path),
                'counts': [11, 100, 500], 'crop': 'feature', 'margin': 64,
                'sum_native_strength': .0003, 'cache_bytes': 256*1024**2,
                'threads': 2, 'wall_budget_s': 900, 'relative_tolerance': 1e-10,
                'scope': 'Resource/operator parity profile only; no converged reconstruction or scientific assessment.',
                'q3_authorized': False}
    directory = open_study(directory, protocol)
    started = time.monotonic()
    rows, failures = [], []
    def timed(fn):
        if time.monotonic()-started >= protocol['wall_budget_s']:
            raise TimeoutError('declared profile budget exhausted')
        start = time.monotonic()
        result = fn()
        return result, time.monotonic()-start
    try:
        data, data_s = timed(lambda: ObservedSceneData(path, range(500)))
        for count in protocol['counts']:
            indices = frame_indices(count)
            ops = data.operators(indices, margin=64)
            images = [data.images[i] for i in indices]
            variances = [data.variances[i] for i in indices]
            ridge = prior_coefficient(.0003, count)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            batch = CellFFTBatch(ops, device='cuda', cache_bytes=protocol['cache_bytes'])
            fast, setup_s = timed(lambda: SceneQuadratic(ops, images, variances, ridge=ridge, batch=batch))
            reference, cpu_setup_s = timed(lambda: SceneQuadratic(ops, images, variances, ridge=ridge))
            x = np.full(fast.shape, max(float(np.mean(images)), 0.)/data.cfg.bin_factor**2)
            cache_before = batch.cache_info()
            a, first_s = timed(lambda: fast.normal(x))
            _, second_s = timed(lambda: fast.normal(x))
            cache_after = batch.cache_info()
            b, reference_s = timed(lambda: reference.normal(x))
            certificate, certificate_s = timed(lambda: reference.certificate(x))
            objective, objective_s = timed(lambda: fast.objective(x))
            cpu_objective, cpu_objective_s = timed(lambda: reference.objective(x))
            errors = {'normal': relative(a, b), 'linear': relative(fast.linear, reference.linear),
                      'objective': abs(objective-cpu_objective)/max(abs(cpu_objective), 1.)}
            row = {'count': count, 'indices': indices, 'passed': max(errors.values()) <= 1e-10,
                   'relative_errors': errors, 'data_psf_setup_s': data_s,
                   'setup_s': setup_s, 'cpu_setup_s': cpu_setup_s,
                   'normal_s': [first_s, second_s], 'cpu_normal_s': reference_s,
                   'cpu_certificate_s': certificate_s, 'probe_certificate': certificate,
                   'objective_s': objective_s, 'cpu_objective_s': cpu_objective_s,
                   'cache_before_normals': cache_before, 'cache_after_normals': cache_after,
                   'gpu_free_before_bytes': free, 'gpu_total_bytes': total,
                   'gpu_name': torch.cuda.get_device_name(),
                   'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'process_peak_rss_bytes': peak_rss_bytes()}
            rows.append(row)
            write_json(directory/f'count-{count}.json', row)
            print(count, 'passed=', row['passed'], 'normal_s=', row['normal_s'],
                  'cache_misses=', cache_after['misses']-cache_before['misses'], flush=True)
            batch.clear_cache()
            del fast, reference, batch, ops
    except Exception as exc:
        failures.append(repr(exc))
        print(repr(exc), flush=True)
    unchanged = identity == identities(deps) and file_hash(path) == protocol['input_sha256']
    passed = len(rows) == 3 and not failures and unchanged and all(r['passed'] for r in rows)
    report = {'status': 'valid' if passed else 'incomplete', 'rows': rows, 'failures': failures,
              'source_input_unchanged': unchanged, 'wall_s': time.monotonic()-started,
              'scope': protocol['scope'], 'q3_authorized': False}
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.exit(0 if run(args.input, args.out)['status'] == 'valid' else 1)
