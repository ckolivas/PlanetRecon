"""Export explicit frequency-taper controls; never changes production stacking.

Preserve frequencies below a stated pass edge, then smoothly taper to zero.
These are diagnostic filters, not an inferred AutoStakkert kernel or a validated
denoising method. Floating arrays retain negative ringing; PNGs use source mapping.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

from alignment_noise_experiment import measurements, read_png


def taper(image, pass_edge, stop_edge):
    if not 0 < pass_edge < stop_edge <= .5:
        raise ValueError('Require 0 < pass edge < stop edge <= 0.5 cycles/pixel')
    # Reflection padding moves the periodic seam well outside the object.
    padded = np.pad(image, 64, mode='reflect')
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    fraction = np.clip((np.hypot(fx, fy)-pass_edge)/(stop_edge-pass_edge), 0, 1)
    transfer = .5*(1+np.cos(np.pi*fraction))
    return np.fft.ifft2(np.fft.fft2(padded)*transfer).real[64:-64, 64:-64]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    with np.load(args.experiment/'local.npz') as data:
        image = data['image'].copy()
    _, metadata = read_png(args.experiment/'local.png')
    mapping = json.loads(metadata)['mapping']
    assert mapping['gamma'] == 1
    report = {}
    for pass_edge, stop_edge in [(.20, .35), (.25, .35)]:
        name = f'local_taper_{round(pass_edge*100):02d}_{round(stop_edge*100):02d}_control'
        filtered = taper(image, pass_edge, stop_edge)
        np.save(args.experiment/f'{name}.npy', filtered)
        pixels = np.ascontiguousarray(np.rint(np.clip(
            (filtered-mapping['black'])/(mapping['white']-mapping['black']), 0, 1)*65535), dtype=np.uint16)
        output = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                        QImage.Format.Format_Grayscale16).copy()
        entry = dict(pass_edge=pass_edge, stop_edge=stop_edge, mapping=mapping,
                     purpose='Cosine frequency-taper diagnostic; ringing and detail loss need assessment',
                     measurements=measurements(filtered))
        output.setText('Experiment', json.dumps(entry))
        if not output.save(str(args.experiment/f'{name}.png')):
            raise OSError(f'Could not save {name}')
        report[name] = entry
    (args.experiment/'frequency_controls.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
