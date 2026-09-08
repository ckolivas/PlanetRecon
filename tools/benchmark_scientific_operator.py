"""Measure isolated CPU phase-gradient work units; never launches Q3."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(n_diam):
    import numpy as np
    from planetrecon.config import make_config
    from planetrecon.mfbd import PupilForward, frame_loss_and_grad
    from planetrecon.physics_audit import peak_rss_bytes
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(2)
    cfg = make_config(1001, 4., n_diam=n_diam, pupil_pad_factor=8., eval_size=2*n_diam)
    t0 = time.monotonic()
    fwd = PupilForward.from_config(cfg)
    setup = time.monotonic()-t0
    n = cfg.eval_size
    y, x = np.indices((n, n))
    obj_f = np.fft.fft2(50.+500.*np.exp(-((x-.45*n)**2+(y-.52*n)**2)/(.15*n)**2))
    alpha = np.random.default_rng(1001).normal(size=60)
    alpha *= .5*np.sqrt(fwd.mask.sum())/np.linalg.norm(alpha)
    image = np.fft.ifft2(fwd.otf(np.zeros(60))*obj_f).real
    rows = []
    for modes in (15, 35, 60):
        frame_loss_and_grad(alpha[:modes], fwd, obj_f, image, 800.)
        samples = []
        for _ in range(20):
            t0 = time.monotonic()
            frame_loss_and_grad(alpha[:modes], fwd, obj_f, image, 800.)
            samples.append(time.monotonic()-t0)
        rows.append({'modes': modes, 'median_s': float(np.median(samples)),
                     'min_s': min(samples), 'max_s': max(samples), 'samples': len(samples)})
    return {'n_diam': n_diam, 'pupil_grid': int(fwd.amplitude.shape[0]), 'detector_grid': n,
            'setup_s': setup, 'gradient': rows, 'process_peak_rss_bytes': peak_rss_bytes(),
            'cpu_threads': 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--measure', type=int)
    args = parser.parse_args()
    if args.measure:
        print(json.dumps(measure(args.measure)))
        return
    if args.out is None:
        parser.error('--out is required')
    from planetrecon.provenance import source_hash, sha256_bytes
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {'q3_authorized': False, 'n_diam': [16, 32, 64], 'modes': [15, 35, 60],
                'gradient_repeats': 20, 'cpu_threads': 2, 'separate_process_per_grid': True,
                'forecast_frames': [500, 1000, 5000, 20000], 'gradient_calls_per_frame_stage': [10, 100, 1000],
                'limits': 'Measured CPU gradient kernels only; work-unit forecasts exclude optimizer variability, object updates, I/O and parallel efficiency. Not an end-to-end or GPU benchmark.'}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for n in protocol['n_diam']:
        result = subprocess.run([sys.executable, __file__, '--measure', str(n)], capture_output=True,
                                text=True, check=True, timeout=120)
        rows.append(json.loads(result.stdout))
    per_cycle = sum(r['median_s'] for r in rows[-1]['gradient'])
    forecasts = [{'frames': n, 'gradient_calls_per_frame_stage': calls,
                  'cpu_work_unit_hours': n*calls*per_cycle/3600.}
                 for n in protocol['forecast_frames'] for calls in protocol['gradient_calls_per_frame_stage']]
    report = {'status': 'diagnostic', 'q3_authorized': False, 'source_hash': source_hash(),
              'benchmark_script_sha256': sha256_bytes(Path(__file__).read_bytes()),
              'measurements': rows, 'forecasts': forecasts}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
