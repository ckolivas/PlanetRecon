"""Measure exact PlanetaryTools sharpened synthetic stacks against their oracle."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.validate_local_warp_centring import image_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    report = json.loads((out/'report.json').read_text())
    report['sharpening'] = json.loads((out/'sharpened'/'sharpening_report.json').read_text())
    for scene, cases in report['scenes'].items():
        for name, case in cases.items():
            stem = f'{scene}_{name}'
            with np.load(out/f'{stem}.npz') as data:
                mask = data['mask']
            with np.load(out/'sharpened'/f'{stem}_noiseless_oracle_stages.npz') as data:
                target = data['sharpened'].mean(axis=2)
            for variant, result in case['variants'].items():
                with np.load(out/'sharpened'/f'{stem}_{variant}_stages.npz') as data:
                    image = data['sharpened'].mean(axis=2)
                # The PT stage is in linear 0..1 units, not camera ADU.
                values = image_errors(image, target, mask)
                result['sharpened_vs_noiseless_oracle'] = {
                    'rmse_linear_0_1': values['rmse_adu'],
                    'gradient_vector_rmse_linear_per_px': values['gradient_vector_rmse_adu_per_px']}
    report['limits'].append('Sharpened errors compare the same two-step nonlinear recipe with the noiseless oracle; they are not noise-only estimates.')
    (out/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    centred_case = ('motion_clean' if 'motion_clean' in report['scenes'].get('saturn', {})
                    else 'motion_blur_noise')
    rows = [(case, label) for case, label in [
        (centred_case, 'Motion centred on reference'),
        ('offset_motion_blur', 'Motion offset from reference, blur + noise')]
        if case in report['scenes'].get('saturn', {})]
    if not rows:
        return
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QFont, QColor
    app = QGuiApplication.instance() or QGuiApplication([])
    canvas = QImage(1500, 290*len(rows), QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setFont(QFont('Sans', 12))
    painter.setPen(QColor('white'))
    local_title = ('Blur-matched template (known blur)' if report.get('matching_template_uses_known_blur')
                   else 'Current local alignment')
    if report.get('coherent_spacing_px') is not None:
        local_title = 'Experimental coherent alignment'
    if report.get('independent_pixel_validation'):
        local_title = 'Independent-pixel validated alignment'
    for row, (case, label) in enumerate(rows):
        for col, (variant, title) in enumerate([
                ('local', local_title), ('centred', 'Subtract mean warp'),
                ('oracle', 'Known true motion')]):
            path = out/'sharpened'/f'saturn_{case}_{variant}_sharpened.png'
            image = QImage(str(path))
            if image.isNull():
                raise ValueError(path)
            px, py = col*500, row*290
            painter.drawText(px+8, py+22, label)
            painter.drawText(px+8, py+45, title)
            painter.drawImage(px, py+58, image.copy(100, 90, 500, 225))
    painter.end()
    if not canvas.save(str(out/'comparison.png')):
        raise OSError('comparison.png')


if __name__ == '__main__':
    main()
