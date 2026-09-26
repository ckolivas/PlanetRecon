"""Measure the controlled stacks and supplied AutoStakkert PNGs in native pixels."""
import argparse
import json
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from scipy.ndimage import center_of_mass, gaussian_filter

from alignment_noise_experiment import measurements, read_png


def bands(image):
    cy, cx = np.round(center_of_mass(np.maximum(image-image.max()*.02, 0))).astype(int)
    output = []
    for lo, hi in [(-62, -22), (30, 70)]:
        roi = image[cy+lo:cy+hi, cx-55:cx+55]
        residual = ((roi-gaussian_filter(roi, 2))/np.median(roi))[4:-4, 4:-4]
        window = np.hanning(residual.shape[0])[:, None]*np.hanning(residual.shape[1])[None, :]
        power = abs(np.fft.fft2(residual*window))**2
        radius = np.hypot(np.fft.fftfreq(residual.shape[0])[:, None],
                          np.fft.fftfreq(residual.shape[1])[None, :])
        output.append([float(power[(radius >= l) & (radius < h)].sum())
                       for l, h in [(.1, .2), (.2, .3), (.3, .4), (.4, .5)]])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--comparison', type=Path, action='append', default=[])
    args = parser.parse_args()
    report = json.loads((args.directory/'report.json').read_text())
    output = {'band_edges_cycles_per_pixel': [.1, .2, .3, .4, .5], 'images': {}}
    for name in report['variants']:
        with np.load(args.directory/f'{name}.npz') as data:
            image = data['image'].copy()
        with np.load(args.directory/f'{name}_halves.npz') as data:
            halves = data['images'].copy()
        output['images'][name] = dict(full=measurements(image),
            split_difference=measurements(image, difference=(halves[0]-halves[1])/2),
            bands=bands(image))
    for path in args.comparison:
        image, _ = read_png(path)
        output['images'][str(path)] = dict(full=measurements(image), bands=bands(image))
    # Diagnostic response controls: these deliberately blur the existing stack.
    # A similar spectrum does not establish AutoStakkert's interpolation kernel.
    with np.load(args.directory/'local.npz') as data:
        image = data['image'].copy()
    output['gaussian_response_controls'] = {}
    for sigma in [.5, .7, .85, 1., 1.1, 1.2]:
        blurred = gaussian_filter(image, sigma)
        output['gaussian_response_controls'][str(sigma)] = dict(
            full=measurements(blurred), bands=bands(blurred))
    _, metadata = read_png(args.directory/'local.png')
    mapping = json.loads(metadata)['mapping']
    blurred = gaussian_filter(image, 1.)
    pixels = np.ascontiguousarray(np.rint(np.clip(
        (blurred-mapping['black'])/(mapping['white']-mapping['black']), 0, 1)*65535),
        dtype=np.uint16)
    preview = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                     QImage.Format.Format_Grayscale16).copy()
    preview.setText('Experiment', json.dumps(dict(
        source='local.npz', gaussian_sigma_pixels=1., mapping=mapping,
        purpose='Deliberately blurred response control, not a production fix')))
    if not preview.save(str(args.directory/'local_sigma1_control.png')):
        raise OSError('Could not save Gaussian response control')
    with np.load(args.directory/'diagnostics.npz') as data:
        q = data['quality'] if report['config']['quality_weighting'] else np.ones_like(data['quality'])
        output['effective_frame_count'] = float(q.sum()**2/(q*q).sum())
        output['residual_field_rms_percentiles_px'] = np.percentile(data['field_rms'], [0, 50, 90, 99, 100]).tolist()
        output['global_shift_range_xy'] = [data['global_shifts'].min(axis=0).tolist(),
                                           data['global_shifts'].max(axis=0).tolist()]
    (args.directory/'analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
