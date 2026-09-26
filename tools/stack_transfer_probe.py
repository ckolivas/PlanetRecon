"""Make exact-copy Saturn captures for measuring stacker spatial transfer.

Run with PYTHONPATH=. .venv/bin/python tools/stack_transfer_probe.py.
No invented seeing, added noise, interpolation, or brightness adjustment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from planetrecon.io.ser import SERSource, write_ser


def integer_translate(frame, dy, dx):
    """Move observed pixels without interpolation; zero-fill newly exposed edges."""
    h, w = frame.shape
    if abs(dy) >= h or abs(dx) >= w:
        raise ValueError('Shift must be smaller than the image')
    result = np.zeros_like(frame)
    result[max(0, dy):min(h, h+dy), max(0, dx):min(w, w+dx)] = frame[
        max(0, -dy):min(h, h-dy), max(0, -dx):min(w, w-dx)]
    return result


def probe_shifts():
    """256 frames, balanced -3..3px shifts, unshifted first frame."""
    grid = np.array([(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)])
    rest = np.concatenate([np.tile(grid, (5, 1)), np.zeros((10, 2), int)])
    np.random.default_rng(20260927).shuffle(rest)
    return np.vstack([np.zeros((1, 2), int), rest])


def verify_capture(path, truth, shifts):
    """Read every serialized frame and undo its known motion on common support."""
    border = int(np.abs(shifts).max()) + 1
    roi = np.s_[border:-border, border:-border]
    total = np.zeros_like(truth, dtype=np.float64)
    with SERSource(path) as source:
        assert source.n_frames() == len(shifts)
        assert source.metadata().bit_depth == 8
        for i, (dy, dx) in enumerate(shifts):
            actual = source.read_raw(i)
            np.testing.assert_array_equal(actual, integer_translate(truth, dy, dx))
            restored = integer_translate(actual, -dy, -dx)
            np.testing.assert_array_equal(restored[roi], truth[roi])
            total += restored
    error = float(np.max(np.abs(total[roi]/len(shifts)-truth[roi])))
    assert error == 0.
    return {'verified_frames': len(shifts), 'common_support_border_px': border,
            'known_shift_mean_max_error_adu': error,
            'ser_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def run_pr(path, out, truth, threads):
    from planetrecon.pipeline.baseline import stack_source
    from planetrecon.reconstruction import ReconstructionConfig
    from planetrecon.result import save_snapshot
    from planetrecon.export import ExportConfig, export_result
    from planetrecon.runtime import apply_thread_limits

    apply_thread_limits(threads)
    config = ReconstructionConfig(device='cpu', threads=threads,
        frame_preselection=False, quality_weighting=False, local_alignment=False,
        reject_saturated=False)
    with SERSource(path) as source:
        result = stack_source(source, config)
    assert not result.incomplete and result.n_used == 256 and result.n_rejected == 0
    save_snapshot(out/f'{path.stem}_pr_global.npz', result)
    export_result(result, out/f'{path.stem}_pr_global.png',
                  ExportConfig(encoding='png16', black=0., white=255., display_gamma=1.))
    # Exclude exposed edge pixels; no registration or photometric fit is hidden here.
    delta = (result.image-truth)[8:-8, 8:-8]
    return {'n_used': result.n_used, 'backend': result.backend,
            'rms_error_adu_interior': float(np.sqrt(np.mean(delta**2))),
            'max_error_adu_interior': float(np.max(np.abs(delta))),
            'warnings': result.warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=Path('2024-09-27-1154_3-CK-R-Sat.ser'))
    parser.add_argument('--out', type=Path, default=Path('out/saturn-transfer-probe'))
    parser.add_argument('--frame', type=int, default=13240,
                        help='Zero-based source frame; default is the previously tried near-midpoint anchor')
    parser.add_argument('--run-pr', action='store_true')
    parser.add_argument('--threads', type=int, default=16, choices=range(1, 33))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with SERSource(args.capture) as source:
        if source.color_mode() != 'mono' or source.metadata().bit_depth != 8:
            raise ValueError('This controlled probe requires an 8-bit mono source')
        truth = source.read_raw(args.frame).astype(np.uint8)
        source_count = source.n_frames()
    args.out.mkdir(parents=True)
    np.save(args.out/'truth_adu.npy', truth)
    # Fixed code expansion, not image-dependent brightness normalization.
    tifffile.imwrite(args.out/'truth_16bit.tif', truth.astype(np.uint16)*257,
                     photometric='minisblack')
    movement = {'static': np.zeros((256, 2), int), 'integer_motion': probe_shifts()}
    report = {'source': str(args.capture.resolve()), 'source_frame_zero_based': args.frame,
              'source_frame_count': source_count, 'frame_count': 256,
              'shape': list(truth.shape),
              'truth_sha256': hashlib.sha256(truth.tobytes()).hexdigest(),
              'truth_tiff_mapping': 'uint16 code = original 8-bit ADU * 257',
              'normalization': False, 'interpolation_in_input': False,
              'AS_results': 'pending external run', 'captures': {}}
    for name, shifts in movement.items():
        path = args.out/f'{name}.ser'
        frames = np.stack([integer_translate(truth, dy, dx) for dy, dx in shifts])
        write_ser(path, frames, instrument='Exact-copy stack transfer probe')
        del frames
        np.save(args.out/f'{name}_shifts_dy_dx.npy', shifts)
        detail = verify_capture(path, truth, shifts)
        if args.run_pr:
            detail['PR_global'] = run_pr(path, args.out, truth, args.threads)
        report['captures'][name] = detail
        print(name, json.dumps(detail), flush=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
