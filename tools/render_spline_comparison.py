"""Render fixed-scale comparisons of the previous and corrected joint fits."""
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QColor, QFont, QImage, QPainter


def main():
    app = QGuiApplication.instance() or QGuiApplication([])
    canvas = QImage(1500, 1580, QImage.Format.Format_RGB32)
    canvas.fill(QColor('black'))
    painter = QPainter(canvas)
    painter.setPen(QColor('white'))
    painter.setFont(QFont('Sans', 15))
    for column, label in enumerate(('Previous joint fit', 'Corrected spline fit', 'Known motion; no added noise')):
        painter.drawText(500*column+12, 28, label)
    rows = [('saturn', 'motion_blur_noise', 'Saturn control: local motion'),
            ('saturn', 'offset_motion_blur', 'Saturn control: offset local motion'),
            ('texture', 'motion_blur_noise', 'Texture control: local motion'),
            ('resolved', None, 'Independent continuous scene: resolved features (2x display)'),
            ('fine', None, 'Independent continuous scene: fine features (2x display)')]
    for row, (scene, condition, label) in enumerate(rows):
        top = 45+row*305
        painter.drawText(12, top+20, label)
        if condition:
            prefix = f'{scene}_{condition}'
            old = Path('out')/f'joint-fit-{scene}-controls/sharpened'
            new = Path('out')/f'joint-spline-{scene}-controls/sharpened'
            paths = [old/f'{prefix}_joint_strong_sharpened.png', new/f'{prefix}_joint_strong_sharpened.png',
                     new/f'{prefix}_noiseless_oracle_sharpened.png']
        else:
            root = Path('out/joint-analytic-motion/sharpened')
            paths = [root/f'{scene}_{method}_sharpened.png' for method in ('bicubic', 'spline', 'noiseless_oracle')]
        for column, path in enumerate(paths):
            image = QImage(str(path))
            if image.isNull():
                raise ValueError(f'Missing image: {path}')
            if condition:
                image = image.copy((image.width()-500)//2, (image.height()-270)//2, 500, 270)
            else:
                image = image.scaled(image.width()*2, image.height()*2,
                                     Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            painter.drawImage(column*500+(500-image.width())//2, top+32, image)
    painter.end()
    output = Path('out/joint-spline-comparison.png')
    if not canvas.save(str(output)):
        raise OSError(output)
    print(output)


if __name__ == '__main__':
    main()
