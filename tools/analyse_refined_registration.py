"""Compare the four registration controls after identical AS smoothing/sharpening."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from PySide6.QtGui import QImage, QPainter, QColor, QFont, QGuiApplication
from PySide6.QtCore import Qt

from tools.alignment_noise_experiment import read_png, measurements
from tools.refined_registration_experiment import NAMES


def ring_edges(image):
    """Descriptive upper-ansa 10–90% transition, not calibrated resolution.

    Average 30 columns on each side, then measure the rise from outer background
    to the maximum before the ring centre. Fixed windows avoid choosing a best
    patch per algorithm. The planet geometry itself also contributes to width.
    """
    cy, cx = np.round(center_of_mass(np.maximum(image-image.max()*.02, 0))).astype(int)
    result = {}
    for name, (lo, hi) in [('left', (-190, -160)), ('right', (160, 190))]:
        profile = image[cy-30:cy+1, cx+lo:cx+hi].mean(axis=1)
        background = np.median(profile[:8])
        peak = int(np.argmax(profile))
        crossings = []
        for fraction in (.1, .9):
            level = background+fraction*(profile[peak]-background)
            index = int(np.flatnonzero(profile[:peak+1] >= level)[0])
            if index == 0:
                raise ValueError('Ring edge is outside fixed comparison window')
            crossings.append(index-1+(level-profile[index-1])/(profile[index]-profile[index-1]))
        result[name] = {'width_10_90_px': float(crossings[1]-crossings[0]),
                        'profile': profile.tolist(), 'crossings': crossings}
    return result


def edge_sensitivity(images):
    """Repeat local/global comparisons over 75 paired windows per ansa.

    Extend peak search through the central ring to avoid a peak clipped at y=0.
    This is a robustness check on a descriptive width, not 75 independent tests.
    """
    centres = {name: np.round(center_of_mass(np.maximum(a-a.max()*.02, 0))).astype(int)
               for name, a in images.items()}
    result = {}
    for side, sign in [('left', -1), ('right', 1)]:
        differences, as_differences = [], []
        for oy in range(-2, 3):
            for ox in range(-2, 3):
                for distance in (165, 175, 185):
                    widths = {}
                    for name, image in images.items():
                        cy, cx = centres[name]+[oy, ox]
                        lo, hi = sign*distance-15, sign*distance+15
                        profile = image[cy-35:cy+6, cx+lo:cx+hi].mean(axis=1)
                        background, peak = np.median(profile[:8]), int(np.argmax(profile))
                        crossings = []
                        for f in (.1, .9):
                            level = background+f*(profile[peak]-background)
                            k = int(np.flatnonzero(profile[:peak+1] >= level)[0])
                            if not k:
                                raise ValueError('Clipped sensitivity window')
                            crossings.append(k-1+(level-profile[k-1])/(profile[k]-profile[k-1]))
                        widths[name] = crossings[1]-crossings[0]
                    differences.append(widths['production_local']-widths['existing_bilinear'])
                    as_differences.append(widths['existing_bilinear']-widths['AS_manual64'])
        result[side] = {'paired_windows': len(differences),
            'local_minus_global_width_px_min_median_max': np.percentile(differences, [0,50,100]).tolist(),
            'fraction_local_wider': float(np.mean(np.array(differences) > 0)),
            'global_minus_AS_width_px_min_median_max': np.percentile(as_differences, [0,50,100]).tolist()}
    return result


def main():
    out = Path('out/saturn-refined-registration')
    report = json.loads((out/'report.json').read_text())
    report['sharpening'] = json.loads((out/'sharpened'/'sharpening_report.json').read_text())
    inputs = {
        'production_local': Path('out/saturn-matched-transfer/sharpened/pr_original_as_transfer_sharpened.png'),
        'AS_manual64': Path('out/saturn-matched-transfer/sharpened/as_manual64_sharpened.png'),
        **{name: out/'sharpened'/f'{name}_as_transfer_sharpened.png' for name in NAMES}}
    report['comparison'] = {}
    arrays = {}
    app = QGuiApplication.instance() or QGuiApplication([])
    canvas = QImage(2000, 1530, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 18))
    labels = {'production_local': 'PR production local + AS smoothing',
              'AS_manual64': 'AutoStakkert manual-64 reference',
              'existing_bilinear': 'Existing shifts / bilinear',
              'existing_cubic': 'Existing shifts / cubic',
              'refined_bilinear': 'Refined shifts / bilinear',
              'refined_cubic': 'Refined shifts / cubic'}
    order = ['production_local', 'AS_manual64', 'existing_bilinear', 'existing_cubic',
             'refined_bilinear', 'refined_cubic']
    for i, name in enumerate(order):
        image, _ = read_png(inputs[name])
        arrays[name] = image
        report['comparison'][name] = {'path': str(inputs[name]), 'sharpened': measurements(image),
                                      'ring_edges': ring_edges(image)}
        cy, cx = np.round(center_of_mass(np.maximum(image-image.max()*.02, 0))).astype(int)
        crop = QImage(str(inputs[name])).copy(int(cx-250), int(cy-115), 500, 230)
        crop = crop.scaled(1000, 460, Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.FastTransformation)
        x, y = (i % 2)*1000, (i//2)*510
        painter.drawText(x+15, y+32, labels[name])
        painter.drawImage(x, y+45, crop)
        print(name, 'variation',*[round(v['highpass_percent'],5) for v in report['comparison'][name]['sharpened'].values()],
              'edge widths',*[round(v['width_10_90_px'],5) for v in report['comparison'][name]['ring_edges'].values()])
    painter.end()
    assert canvas.save(str(out/'comparison.png'))
    report['edge_sensitivity'] = edge_sensitivity({name: arrays[name] for name in
        ('production_local', 'existing_bilinear', 'AS_manual64')})
    print('Edge sensitivity:', json.dumps(report['edge_sensitivity']))
    report['limits'] = [
        'Four experiment cells use global alignment only, fixed 64-frame reference and identical 5738 frames/weights.',
        'The production-local and AS outputs are external comparison controls, not members of the factorial experiment.',
        'Every PR comparison has sigma-1 radius-3 smoothing and the same sharpening; no brightness normalization was added.',
        'Cubic preserves more high-frequency contrast and noise and can introduce overshoot.',
        'Ansa transition widths include physical geometry, seeing, interpolation and sharpening; not calibrated resolution.',
        'The smoothing model was measured on supplied AS controls, not extracted from its private implementation.',
        'No production algorithm or default is changed.']
    (out/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
