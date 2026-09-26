"""Diagnostic AS-measured smoothing on PR copies, with unchanged export mapping.

Prepare inputs, run planetarytools_sharpen_experiment.py on the five PNGs, then
run this tool with --analyse. No production stacking defaults are modified.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter

from planetrecon.export import ExportConfig, export_result
from planetrecon.result import load_snapshot, save_snapshot
from tools.alignment_noise_experiment import measurements, read_png


INPUTS = {
    'pr_original': Path('2024-09-27-1154_3-CK-R-SatP.png'),
    'pr_multiscale': Path('out/saturn-multiscale-patches/multiscale_local2_registered.png'),
    'as_manual64': Path('AS_F5738/2024-09-27-1154_3-CK-R-Sat_r64_lapl4_ap54.png'),
}
SNAPSHOTS = {
    'pr_original': Path('out/saturn-controlled-alignment/local.npz'),
    'pr_multiscale': Path('out/saturn-multiscale-patches/multiscale_local2_registered.npz'),
}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    report = {'purpose': 'Comparison with AS empirically measured sigma-1 radius-3 smoothing',
              'production_changed': False, 'brightness_normalization': False,
              'inputs': {}, 'variants': {}}
    for name, source in INPUTS.items():
        shutil.copyfile(source, out/f'{name}.png')
        report['inputs'][name] = {'source': str(source),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    for name, path in SNAPSHOTS.items():
        original = load_snapshot(path)
        pixels, text = read_png(INPUTS[name])
        mapping = json.loads(text)['mapping']
        assert mapping['gamma'] == 1.
        predicted = np.floor(65535*np.clip((original.image-mapping['black'])/
                                          (mapping['white']-mapping['black']), 0, 1)+.5)
        np.testing.assert_array_equal(predicted, pixels)
        result = deepcopy(original)
        result.image = gaussian_filter(original.image, 1., radius=3)
        result.provenance['comparison_smoothing'] = {'sigma_px': 1., 'radius_px': 3,
            'basis': 'measured AS exact-copy probe', 'diagnostic_only': True}
        variant = name+'_as_transfer'
        save_snapshot(out/f'{variant}.npz', result)
        export_result(result, out/f'{variant}.png', ExportConfig(encoding='png16',
            black=mapping['black'], white=mapping['white'], display_gamma=1.))
        report['variants'][variant] = {'snapshot': str(path), 'mapping': mapping,
            'unmodified_snapshot_reproduces_original_pixels': True}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def montage(out):
    from PySide6.QtGui import QImage, QPainter, QColor, QFont, QGuiApplication
    from PySide6.QtCore import Qt
    app = QGuiApplication.instance() or QGuiApplication([])
    panels = [('pr_original', 'PR original'),
              ('pr_original_as_transfer', 'PR + measured AS smoothing'),
              ('as_manual64', 'AutoStakkert, manual 64-frame reference'),
              ('pr_multiscale_as_transfer', 'PR experimental multiscale + AS smoothing')]
    canvas = QImage(2000, 1020, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 18))
    for i, (name, title) in enumerate(panels):
        path = out/'sharpened'/f'{name}_sharpened.png'
        data, _ = read_png(path)
        cy, cx = np.round(center_of_mass(np.maximum(data-data.max()*.02, 0))).astype(int)
        image = QImage(str(path)).copy(int(cx-250), int(cy-115), 500, 230)
        image = image.scaled(1000, 460, Qt.AspectRatioMode.IgnoreAspectRatio,
                             Qt.TransformationMode.FastTransformation)
        x, y = (i % 2)*1000, (i//2)*510
        painter.drawText(x+15, y+32, title)
        painter.drawImage(x, y+45, image)
    painter.end()
    assert canvas.save(str(out/'comparison.png'))


def analyse(out):
    report = json.loads((out/'report.json').read_text())
    report['sharpening'] = json.loads((out/'sharpened'/'sharpening_report.json').read_text())
    report['measurements'] = {}
    for name in [*INPUTS, *report['variants']]:
        raw, _ = read_png(out/f'{name}.png')
        sharp, _ = read_png(out/'sharpened'/f'{name}_sharpened.png')
        values = {'raw': measurements(raw), 'sharpened': measurements(sharp)}
        report['measurements'][name] = values
        print(name, {k: v['highpass_percent'] for k, v in values['sharpened'].items()})
    report['limits'] = [
        'Measured AS response came from identical-frame controls, not these real-data stacks.',
        'This applies a comparison blur to PR copies; production algorithms are unchanged.',
        'AS and production PR use 5738 frames; experimental PR selects 5738 separately per AP.',
        'No new brightness normalization; each existing input retains its original export mapping.',
        'Disc fine-scale variation includes detail and artifacts, not just sensor noise.',
        'Montage uses rounded-centroid crops and nearest-neighbor 2x display, no subpixel registration.',
        'Visual sharpness is not a calibrated resolution measurement.']
    (out/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    montage(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('out/saturn-matched-transfer'))
    parser.add_argument('--analyse', action='store_true')
    args = parser.parse_args()
    (analyse if args.analyse else prepare)(args.out)
