"""Measure broad-band Fourier agreement of two Saturn stack halves.

This is a repeatability diagnostic, not a calibrated resolution estimate.
Shared systematic errors can correlate; changing seeing/registration can lower
correlation. No noise source is inferred from a low correlation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, fourier_shift, gaussian_filter, shift
from scipy.optimize import least_squares

from planetrecon.backends.cpu import CPUBackend


BANDS = [(.05, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5)]
REGIONS = {'upper_disc': (-62, -22, -55, 55),
           'lower_disc': (30, 70, -55, 55),
           'left_ring': (-20, 20, -215, -100),
           'right_ring': (-20, 20, 100, 215),
           'north_limb': (-110, -65, -70, 70)}


def align_pair(first, second):
    """Estimate translation from blurred proxies, shift final image in Fourier space."""
    a, b = gaussian_filter(first, 3), gaussian_filter(second, 3)
    xy = CPUBackend(threads=8).phase_correlation(a, b)
    initial_yx = -np.asarray(xy)[::-1]
    cy, cx = np.round(center_of_mass(np.maximum(a-a.max()*.02, 0))).astype(int)
    section = np.s_[cy-115:cy+115, cx-240:cx+240]

    def residual(yx):
        moved = shift(b, yx, order=1, prefilter=False)
        return ((moved-a)[section]/a.max()).ravel()

    fitted = least_squares(residual, initial_yx, bounds=(initial_yx-1, initial_yx+1))
    if not fitted.success:
        raise RuntimeError(fitted.message)
    aligned = np.fft.ifft2(fourier_shift(np.fft.fft2(second), fitted.x)).real
    return aligned, dict(shift_yx=fitted.x.tolist(),
        proxy_rms_before=float(np.sqrt(np.mean(residual([0, 0])**2))),
        proxy_rms_after=float(np.sqrt(np.mean(residual(fitted.x)**2))))


def analyse_pair(first, second):
    mean = (first+second)/2
    cy, cx = np.round(center_of_mass(np.maximum(mean-mean.max()*.02, 0))).astype(int)
    report = {}
    for name, (y0, y1, x0, x1) in REGIONS.items():
        section = np.s_[cy+y0:cy+y1, cx+x0:cx+x1]
        patches = np.array([image[section] for image in (first, second)])
        residuals = np.array([p-gaussian_filter(p, 2) for p in patches])[:, 4:-4, 4:-4]
        height, width = residuals.shape[1:]
        window = np.hanning(height)[:, None]*np.hanning(width)[None, :]
        spectra = np.fft.fft2(residuals*window)
        radius = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.fftfreq(width)[None, :])
        rows = []
        for lo, hi in BANDS:
            mask = (radius >= lo) & (radius < hi)
            a, b = spectra[:, mask]
            aa, bb = np.sum(abs(a)**2), np.sum(abs(b)**2)
            cross = np.sum((a*b.conj()).real)
            average_power = .25*(aa+bb+2*cross)
            difference_power = .25*(aa+bb-2*cross)
            rows.append(dict(band=[lo, hi], correlation=float(cross/np.sqrt(aa*bb)),
                difference_to_mean_power=float(difference_power/average_power),
                fourier_samples=int(mask.sum())))
        report[name] = dict(bounds_yxyx=[int(v) for v in (cy+y0, cx+x0, cy+y1, cx+x1)], bands=rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.shared/'local_halves.npz') as data:
        original = data['images'].copy()
    separate = []
    for i in range(2):
        with np.load(args.independent/f'half{i}.npz') as data:
            separate.append(data['image'].copy())
    aligned, registration = align_pair(*separate)
    output = dict(independent=analyse_pair(separate[0], aligned),
        independent_before_translation=analyse_pair(*separate),
        shared_reference=analyse_pair(*original), registration=registration,
        notes=['No formal resolution threshold or independence confidence interval is inferred.',
               'Independent runs have disjoint selected frames and reference frames.',
               'Seeing, reference shapes and common detector patterns remain possible confounds.',
               'Difference/mean power near one means little common structure in that band.',
               'Image 2 is translated once in Fourier space; no Gaussian applied to measured images.'])
    (args.independent/'repeatability.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(registration))
    for name, region in output['independent'].items():
        print(name, [round(row['correlation'], 3) for row in region['bands']])


if __name__ == '__main__':
    main()
