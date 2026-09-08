"""Bounded W02 centroid calibration over archived development displacement ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from planetrecon.config import make_config
from planetrecon.mfbd import PupilForward, tip_tilt_from_shifts
from planetrecon.optics import centroid_px
from planetrecon.runtime import apply_thread_limits


def run(output):
    apply_thread_limits(2)
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'role': 'bounded development diagnostic', 'q3_authorized': False,
                'grid_points_per_axis': 3, 'higher_mode_phase_rms_rad': [0., .5, 1.],
                'centroid_tolerance_px': 1e-6, 'max_nfev': 64,
                'cases': 'all 12 archived reduced development cases; recorded min/max displacement rectangles',
                'limitations': 'Centroid calibration does not establish physical pupil tilt from image motion or correct exposure/basis/model mismatch.'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    records = []
    for path in sorted((root/'results/r10-development-audit').glob('audit-*.json')):
        raw = path.read_bytes()
        case = json.loads(raw)
        c = case['config']
        cfg = make_config(case['seed'], case['dr0'], n_diam=c['n_diam'],
                          pupil_pad_factor=c['pupil_pad_factor'], eval_size=c['eval_size'])
        fwd = PupilForward.from_config(cfg)
        low, high = np.array(case['shift_range_px'])
        targets = np.array([(x, y) for x in np.linspace(low[0], high[0], 3)
                           for y in np.linspace(low[1], high[1], 3)])
        zero = np.array(centroid_px(fwd.psf_det(np.zeros(2))))
        legacy_jac = np.column_stack([np.array(centroid_px(fwd.psf_det(v)))-zero for v in np.eye(2)])
        legacy = np.array([np.linalg.lstsq(legacy_jac, v-zero, rcond=None)[0] for v in targets])
        rng = np.random.default_rng(case['seed'])
        base = rng.normal(size=fwd.n_modes)
        base[:2] = 0.
        base /= np.std(fwd.phase(base)[fwd.mask])
        for rms in protocol['higher_mode_phase_rms_rad']:
            phases = np.tile(base*rms, (len(targets), 1))
            previous = phases.copy()
            previous[:, :2] = legacy
            legacy_error = np.linalg.norm(np.array([centroid_px(fwd.psf_det(a)) for a in previous])-targets, axis=1)
            row = {'case': path.name, 'input_sha256': hashlib.sha256(raw).hexdigest(),
                   'higher_mode_phase_rms_rad': rms, 'n_targets': len(targets),
                   'legacy_max_centroid_error_px': float(legacy_error.max())}
            try:
                coefficients, info = tip_tilt_from_shifts(fwd, targets, phase_base=phases, return_info=True)
                phases[:, :2] = coefficients
                actual = np.array([centroid_px(fwd.psf_det(a)) for a in phases])
                error = float(np.linalg.norm(actual-targets, axis=1).max())
                row.update(status='passed' if error <= 1e-6 else 'failed', max_centroid_error_px=error,
                           max_nfev=max(r['nfev'] for r in info['frames']))
            except ValueError as exc:
                row.update(status='failed', reason=str(exc))
            records.append(row)
    report = {'protocol': protocol, 'records': records, 'wall_time_s': time.monotonic()-started,
              'source_sha256': hashlib.sha256((root/'planetrecon/mfbd.py').read_bytes()).hexdigest(),
              'all_passed': len(records) == 36 and all(r['status'] == 'passed' for r in records)}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({'all_passed': result['all_passed'], 'cases': len(result['records']),
                      'wall_time_s': result['wall_time_s']}, indent=2))
    raise SystemExit(0 if result['all_passed'] else 1)
