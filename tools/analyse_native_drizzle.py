"""Measure the fixed-translation native drizzle control with exact sharpening."""
import json
from pathlib import Path

from tools.alignment_noise_experiment import read_png, measurements
from tools.analyse_refined_registration import ring_edges
from tools.analyse_joint_saturn import sharp_identity
from tools.joint_saturn_experiment import digest


def comparison(root):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import numpy as np
    from scipy.ndimage import center_of_mass
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QFont
    app = QGuiApplication.instance() or QGuiApplication([])
    paths = [root/'sharpened'/f'{name}_sharpened.png' for name in ('square_1', 'square_half', 'point')]
    paths.append(Path('out/saturn-matched-transfer/sharpened/as_manual64_sharpened.png'))
    labels = ['Unit square: identical to global bilinear', 'Half-pixel square footprint',
              'Point placement: identical to integer shifts', 'AutoStakkert: own selection and local alignment']
    canvas = QImage(2000, 1090, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 18))
    for i, (path, label) in enumerate(zip(paths, labels)):
        pixels = read_png(path)[0]
        cy, cx = np.rint(center_of_mass(np.maximum(pixels-.02*pixels.max(), 0))).astype(int)
        image = QImage(str(path)).copy(int(cx-250), int(cy-115), 500, 230)
        image = image.scaled(1000, 460, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        x, y = (i % 2)*1000, (i//2)*510
        painter.drawText(x+12, y+32, label)
        painter.drawImage(x, y+45, image)
    painter.drawText(12, 1060, '5,738 frames; exact user sharpening; deposition controls use fixed global translations')
    painter.end()
    if not canvas.save(str(root/'comparison.png')):
        raise OSError('comparison image')


def main():
    root = Path('out/saturn-native-drizzle')
    report = json.loads((root/'report.json').read_text())
    prior = json.loads(Path('out/saturn-coherent-field/sharpened/sharpening_report.json').read_text())
    report['sharpening'] = sharp_identity(root, prior)
    for name, values in report['variants'].items():
        path = root/'sharpened'/f'{name}_sharpened.png'
        pixels = read_png(path)[0]
        values.update(path=str(path), sha256=digest(path), variation=measurements(pixels), ring_edges=ring_edges(pixels))
        print(name, [v['highpass_percent'] for v in values['variation'].values()],
              [v['width_10_90_px'] for v in values['ring_edges'].values()])
    report['limits'] = [
        'Fixed global translations only; no local deformation, AP recombination or resolution enlargement.',
        'Area-overlap square footprints at native scale are explicit test kernels, not claims about AS internals.',
        'A unit square footprint equals bilinear sampling for translations; point placement equals rounded translations.',
        'At exact integer placement these kernels preserve an impulse and cannot generate the previously measured broad AS response.',
        'Fine-scale variation includes detail and artifacts; descriptive ring widths are not calibrated resolution.']
    report['developer_source'] = dict(date='2024-01-18', author='Emil Kraaikamp (MvZ)', post=39,
        url='https://www.cloudynights.com/forums/topic/907378-jupiter-at-20-degrees-11624-not-what-you-think-dont-miss-panel-26/page/2/',
        description='Distinguishes integer regular stacking, subpixel enlarged drizzle and temporal Bayer channel reconstruction; not a current-version implementation audit.')
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    Path('results/registration/native-mono-drizzle.json').write_text(json.dumps(report, indent=2)+'\n')
    comparison(root)


if __name__ == '__main__':
    main()
