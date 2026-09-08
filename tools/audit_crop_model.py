"""Localize noiseless crop-forward mismatch in full-grid development captures."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from tools.audit_full_gate1 import complete_family, file_hash, write_json


def regional_errors(model, expected, observed, variance, window):
    """Squared residual per expected noise variance, with declared spatial weights."""
    weights = {'whole_crop': np.ones_like(window), 'metric_window_squared': window**2}
    for border in (8, 16, 32):
        if 2*border >= window.shape[0]:
            continue
        mask = np.zeros_like(window)
        mask[border:-border, border:-border] = 1.
        weights[f'interior_border_{border}'] = mask
    mismatch = np.mean((model-expected)**2/variance, axis=0)
    noise = np.mean((observed-expected)**2/variance, axis=0)
    return {name: {'effective_pixels': float(w.sum()),
                   'model_mean_standardized_square': float(np.sum(w*mismatch)/w.sum()),
                   'noise_mean_standardized_square': float(np.sum(w*noise)/w.sum())}
            for name, w in weights.items()}


def run(inputs, directory):
    from planetrecon import constants as C
    from planetrecon.evaluate import family_paths, load_crop
    from planetrecon.hdf5io import gate_errors
    from planetrecon.provenance import source_hash
    from planetrecon.runtime import apply_thread_limits
    apply_thread_limits(2)
    directory.mkdir(parents=True, exist_ok=False)
    identity, runner = source_hash(), file_hash(__file__)
    write_json(directory/'protocol.json', {
        'source_hash': identity, 'runner_sha256': runner, 'seeds': C.DEV_SEEDS,
        'regimes': C.MANDATORY_DR0, 'crops': ['feature', 'bland'], 'n_frames': 500,
        'eval_size': 128, 'q3_authorized': False,
        'design': 'Compare deployed crop-circular convolution of detector truth with simulator '
                  'padded optical convolution/integration/cropping expectations. Normalize deterministic '
                  'model discrepancy and realized noise separately by oracle expected+read variance. '
                  'Full crop, squared metric window, and fixed 8/16/32-pixel interior borders. '
                  'Localization only: no fitted alignment, no reconstruction or model acceptance threshold.'})
    started = time.monotonic()
    cases = []
    for path in family_paths(inputs, C.DEV_SEEDS):
        if errors := gate_errors(path):
            raise ValueError(f'{path}: {errors}')
        input_hash = file_hash(path)
        for crop_name in ('feature', 'bland'):
            cfg, crop, extras = load_crop(path, crop_name)
            if cfg.n_frames != 500 or cfg.eval_size != 128:
                raise ValueError('full-resolution 500-frame inputs required')
            model = np.fft.ifft2(crop.otf*np.fft.fft2(crop.truth_e)).real
            variance = np.maximum(crop.expected, 0)+extras['read_noise_e']**2
            row = {'seed': cfg.seed, 'dr0': cfg.dr0, 'crop': crop_name,
                   'input_path': str(path), 'input_sha256': input_hash,
                   'regions': regional_errors(model, crop.expected, crop.observed, variance, crop.window)}
            cases.append(row)
            print(f'{cfg.seed} {cfg.dr0} {crop_name}: interior32={row["regions"]["interior_border_32"]["model_mean_standardized_square"]:.4g}', flush=True)
        if input_hash != file_hash(path):
            raise ValueError(f'input changed during audit: {path}')
    report = {'status': 'diagnostic', 'source_hash': identity,
              'source_unchanged': identity == source_hash() and runner == file_hash(__file__),
              'complete': complete_family(cases, C.DEV_SEEDS, C.MANDATORY_DR0),
              'q3_authorized': False, 'cases': cases, 'wall_s': time.monotonic()-started}
    write_json(directory/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.out)
