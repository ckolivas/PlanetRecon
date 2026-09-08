"""Full detector-grid padding, pupil, exposure and low-frequency controls."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from planetrecon import constants as C
from planetrecon.config import make_config
from planetrecon.provenance import source_hash
from planetrecon.validate import check_lowfreq_tilt, check_padding_and_grid


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {'status': 'diagnostic', 'q3_authorized': False, 'seeds': list(C.DEV_SEEDS),
                'regimes': [4., 8.], 'crops': ['feature', 'bland'], 'eval_size': 128,
                'pupil_samples': [64, 128], 'detector_padding': [64, 128],
                'exposure_quadrature': [8, 16], 'subharmonic_levels': [4, 5],
                'low_frequency_independent_screens': 16,
                'limits': 'Single-frame image quadrature/padding/grid controls and 16-screen low-frequency moments; not a complete reconstruction or gap family.',
                'tolerances': {'padding': C.EH_PADDING_TOL, 'grid': C.EH_GRID_TOL,
                               'exposure': C.EH_EXPOSURE_TOL, 'tilt_grid': C.TILT_GRID_TOL,
                               'strehl_low_frequency': C.STREHL_LOFREQ_TOL, 'tilt_low_frequency': C.TILT_LOFREQ_TOL}}
    (directory/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    rows = []
    for seed in protocol['seeds']:
        for regime in protocol['regimes']:
            cfg = make_config(seed, regime, n_frames=1)
            for crop in protocol['crops']:
                print(f'full-grid physics seed={seed} D/r0={regime} crop={crop}', flush=True)
                t0 = time.monotonic()
                checks = check_padding_and_grid(cfg, crop_name=crop)
                if crop == 'feature':
                    checks += check_lowfreq_tilt(cfg)
                row = {'seed': seed, 'dr0': regime, 'crop': crop, 'checks': [asdict(c) for c in checks],
                       'wall_s': time.monotonic()-t0,
                       'cumulative_process_peak_rss_bytes': peak_rss_bytes()}
                rows.append(row)
                (directory/f'case-{seed}-{int(regime)}-{crop}.json').write_text(json.dumps(row, indent=2)+'\n')
                print(f"  {sum(c.passed for c in checks)}/{len(checks)} pass; {row['wall_s']:.2f}s", flush=True)
    report = {'status': 'diagnostic', 'q3_authorized': False, 'source_hash': source_hash(),
              'cases': rows, 'wall_s': time.monotonic()-started,
              'all_controls_passed': all(c['passed'] for r in rows for c in r['checks'])}
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak)*(1 if sys.platform == 'darwin' else 1024)


if __name__ == '__main__':
    import argparse
    from planetrecon.runtime import apply_thread_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    apply_thread_limits(2)
    run(args.out)
