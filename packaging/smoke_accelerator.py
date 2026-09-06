"""Verify frozen CUDA parity, checkpoint continuation and allocator recovery.

The child runs with Python/venv/executable search paths disabled. This requires
real supported NVIDIA hardware; CPU fallback cannot pass the CUDA cases.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planetrecon.io.ser import write_ser, COLOR_RGGB
from planetrecon.result import load_snapshot


def main(executable, out):
    executable = str(Path(executable).resolve())
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(1010)
    frame = rng.integers(100, 1000, (24, 32), dtype='u2')
    source = write_ser(out / 'fixture.ser', np.stack([np.roll(frame, i, axis=0) for i in range(4)]), color_id=COLOR_RGGB)
    env = {**os.environ, 'PATH': '/nonexistent', 'PYTHONHOME': '/nonexistent',
           'PYTHONPATH': '/nonexistent', 'VIRTUAL_ENV': '/nonexistent'}
    state = out / 'state.npz'
    cases = [('cpu', ['--device', 'cpu']),
             ('gpu', ['--device', 'gpu', '--cuda-memory-mib', '64', '--state-checkpoint', str(state)]),
             ('resume', ['--device', 'gpu', '--cuda-memory-mib', '64', '--resume', str(state)]),
             ('limited', ['--device', 'gpu', '--cuda-memory-mib', '1'])]
    results = {}
    for name, flags in cases:
        with (out / f'{name}.log').open('w') as log:
            subprocess.run([executable, '--threads', '2', 'stack', '--path', str(source),
                            '--out', str(out / name), *flags], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        results[name] = load_snapshot(out / name / 'stack.npz')
    assert results['gpu'].backend == results['resume'].backend == 'cuda'
    assert results['limited'].backend == 'cpu'
    for name, result in results.items():
        np.testing.assert_array_equal(result.validity, results['cpu'].validity)
        np.testing.assert_allclose(result.image, results['cpu'].image, rtol=1e-12, atol=1e-10)
        assert (result.n_used, result.n_rejected, result.incomplete) == (4, 0, False)
        if name != 'cpu':
            budget = result.provenance['cuda_allocation_budget']
            assert budget['enforced']
            assert budget['peak_reserved_bytes'] <= budget['effective_bytes']
    np.testing.assert_array_equal(results['resume'].image, results['gpu'].image)
    report = {'status': 'passed', 'python_paths_disabled': True,
              'gpu_cpu_parity': True, 'completed_state_resume_exact': True,
              'allocator_limit_cpu_recovery': True,
              'cases': {name: {'backend': result.backend, 'n_used': result.n_used,
                              'budget': result.provenance.get('cuda_allocation_budget')}
                        for name, result in results.items()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.executable, args.out)
