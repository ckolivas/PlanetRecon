"""Measure spatial transfer in exact-copy AutoStakkert controls.

Only the input is transformed to predict output pixels. The measured AS image
is never resampled. No fitted model is applied to production stacks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from tools.alignment_noise_experiment import read_png


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(actual, predicted):
    delta = np.asarray(predicted)-actual
    return {'rms_adu': float(np.sqrt(np.mean(delta**2))),
            'maximum_absolute_adu': float(np.max(np.abs(delta))),
            'mean_adu': float(np.mean(delta))}


def coarse_displacement(truth, output):
    corr = fftconvolve(output, truth[::-1, ::-1], mode='full')
    y, x = np.unravel_index(corr.argmax(), corr.shape)
    return np.array([y-truth.shape[0]+1, x-truth.shape[1]+1])


def fit_kernel(truth, output, source_y, source_x, train, test, size=7):
    half = size//2
    patches = sliding_window_view(truth, (size, size))[
        source_y-half, source_x-half].reshape(-1, size*size)
    matrix = np.column_stack([patches, np.ones(len(patches))])
    coefficients = np.linalg.lstsq(matrix[train], output[train], rcond=None)[0]
    kernel = coefficients[:-1].reshape(size, size)
    axis = np.arange(size)-half
    expected = np.exp(-axis**2/2)
    expected /= expected.sum()
    return {'kernel': kernel.tolist(), 'kernel_sum': float(kernel.sum()),
            'offset_adu': float(coefficients[-1]),
            'max_coefficient_difference_from_sigma1_gaussian':
                float(np.max(np.abs(kernel-np.outer(expected, expected)))),
            'held_out': errors(output[test], matrix[test]@coefficients)}


def fit_translation(truth, output, source_y, source_x, train, test):
    # Cubic interpolation with prefilter preserves source samples at integers.
    # Unlike fitting a free kernel, this model permits only translation plus
    # linear code scaling/offset. Fit within the coarse integer peak's cell.
    coefficients = spline_filter(truth, order=3)

    def predict(p, mask):
        dy, dx, gain, offset = p
        return gain*map_coordinates(coefficients,
            [source_y[mask]+dy, source_x[mask]+dx], order=3,
            prefilter=False, mode='constant')+offset

    fitted = least_squares(lambda p: predict(p, train)-output[train],
        [0., 0., 1., 0.], bounds=([-.5, -.5, .5, -20], [.5, .5, 1.5, 20]),
        diff_step=1e-4, max_nfev=80)
    return {'model': 'cubic subpixel pull plus gain and offset',
            'parameters_dy_dx_gain_offset': fitted.x.tolist(),
            'optimizer_success': bool(fitted.success),
            'held_out': errors(output[test], predict(fitted.x, test))}


def analyse(truth, codes):
    output = codes/256.  # AS's measured fixed code expansion; gain fit below checks it.
    displacement = coarse_displacement(truth, output)
    y, x = np.indices(output.shape)
    # Fixed Saturn region; two spatially separate halves for model fit/validation.
    region = (y >= 60)&(y < 340)&(x >= 100)&(x < 600)
    sy, sx = y[region]-displacement[0], x[region]-displacement[1]
    if not ((sy >= 4)&(sy < truth.shape[0]-4)&
            (sx >= 4)&(sx < truth.shape[1]-4)).all():
        raise ValueError('Comparison region lies outside common input support')
    target = output[region]
    train, test = x[region] < 350, x[region] >= 350
    predicted = gaussian_filter(truth, 1., radius=3)[sy, sx]
    identity = truth[sy, sx]
    gaussian_error = errors(target[test], predicted[test])
    original_error = errors(target[test], identity[test])
    return {'shape': list(codes.shape), 'coarse_displacement_dy_dx': displacement.tolist(),
            'output_codes_per_adu': 256,
            'comparison_region_yx': [[60, 340], [100, 600]],
            'training_columns': [100, 350], 'held_out_columns': [350, 600],
            'integer_translation_only_held_out': original_error,
            'fractional_translation': fit_translation(truth, target, sy, sx, train, test),
            'fixed_gaussian_sigma1_radius3_held_out': gaussian_error,
            'fixed_gaussian_all_region': errors(target, predicted),
            'fixed_gaussian_explained_squared_discrepancy_fraction':
                1-(gaussian_error['rms_adu']/original_error['rms_adu'])**2,
            'empirical_7x7': fit_kernel(truth, target, sy, sx, train, test)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, default=Path('out/saturn-transfer-probe'))
    parser.add_argument('--stacks', type=Path, default=Path('stack/AS_P100'))
    parser.add_argument('--out', type=Path, default=Path('out/saturn-transfer-analysis'))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    original = json.loads((args.probe/'report.json').read_text())
    truth = np.load(args.probe/'truth_adu.npy').astype(float)
    report = {'captures': {}, 'scope': 'Empirical transfer of these supplied outputs; processing stage unknown'}
    arrays = {}
    for name in ('static', 'integer_motion'):
        capture = args.stacks.parent/f'{name}.ser'
        actual_hash = digest(capture)
        assert actual_hash == original['captures'][name]['ser_sha256']
        path = args.stacks/f'{name}_lapl4_ap55.png'
        codes, _ = read_png(path)
        arrays[name] = codes
        report['captures'][name] = {'ser_sha256': actual_hash,
            'png_sha256': digest(path), 'png': str(path), **analyse(truth, codes)}
    a, b = arrays['static'], arrays['integer_motion']
    region = np.s_[60:340, 100:600]
    report['pair'] = {'whole_image': errors(a/256, b/256),
                      'whole_image_identical_fraction': float(np.mean(a == b)),
                      'comparison_region': errors(a[region]/256, b[region]/256),
                      'comparison_region_identical_fraction': float(np.mean(a[region] == b[region]))}
    args.out.mkdir(parents=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
